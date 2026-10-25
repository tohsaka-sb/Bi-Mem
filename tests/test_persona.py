import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from offline_memory import system
from test_incremental import fact, scene


def source(scene_id, date, content='source scene'):
    return {'scene_id': scene_id, 'content': content,
            'timestamp_start': date, 'timestamp_end': date}


def claim(claim_id, content, date=None):
    return {'id': claim_id, 'content': content,
            'evidence': [source('archived-' + claim_id, date)] if date else []}


def persona(memory, claims):
    memory.graph.add_node('p', level='persona', attribute_type='interests',
                          content='\n'.join(c['content'] for c in claims),
                          timestamp='2099-01-01', persona_claims=claims)


def new_scene(memory, node_id='incoming', date='2025-01-01'):
    fact(memory, node_id + '-fact', timestamp=date)
    scene(memory, node_id, [node_id + '-fact'], timestamp='2100-01-01')


def decide(memory, scene_id, relation, content='new preference', targets=(), dimension='interests'):
    response = {'decisions': [{'dimension': dimension, 'relation': relation,
                              'content': content, 'target_claim_ids': list(targets)}]}
    with patch.object(memory.llm_controller.llm, 'get_completion', return_value=json.dumps(response)):
        memory._update_persona_incrementally([scene_id])


class PersonaTests(unittest.TestCase):
    def test_compatible_older_scene_adds_without_recency_gate(self):
        memory = system()
        persona(memory, [claim('old', 'likes music', '2026-01-01')])
        memory.graph.add_node('values', level='persona', attribute_type='values', content='unchanged')
        new_scene(memory, date='1900-01-01')
        decide(memory, 'incoming', 'adds', content='also enjoys painting')
        self.assertEqual(memory.graph.nodes['p']['content'], 'likes music\nalso enjoys painting')
        self.assertEqual(memory.graph.nodes['values']['content'], 'unchanged')
        self.assertTrue(memory.graph.has_edge('p', 'incoming'))

    def test_conflicting_newer_scene_replaces_only_targeted_claim(self):
        memory = system()
        persona(memory, [claim('food', 'likes spicy food', '2020-01-01'),
                         claim('music', 'plays piano', '2030-01-01')])
        new_scene(memory, date='2025-01-01')
        decide(memory, 'incoming', 'conflicts', content='prefers mild food', targets=['food'])
        attrs = memory.graph.nodes['p']
        self.assertEqual(attrs['content'], 'plays piano\nprefers mild food')
        self.assertEqual(attrs['persona_claims'][0]['id'], 'music')
        record = attrs['persona_conflicts'][0]
        self.assertEqual(record['resolution'], 'replaced')
        self.assertEqual(record['previous_claims'][0]['id'], 'food')
        self.assertEqual(record['previous_evidence_time'], '2020-01-01T00:00:00+00:00')
        self.assertEqual(record['incoming_evidence_time'], '2025-01-01T00:00:00+00:00')

    def test_older_conflicting_scene_cannot_overwrite_newer_evidence(self):
        memory = system()
        persona(memory, [claim('food', 'prefers mild food', '2025-01-01')])
        new_scene(memory, date='2020-01-01')
        decide(memory, 'incoming', 'conflicts', content='likes spicy food', targets=['food'])
        self.assertEqual(memory.graph.nodes['p']['content'], 'prefers mild food')
        self.assertEqual(memory.graph.nodes['p']['persona_conflicts'][0]['resolution'], 'retained')
        self.assertFalse(memory.graph.has_edge('p', 'incoming'))

    def test_equal_evidence_times_stay_unresolved(self):
        memory = system()
        persona(memory, [claim('old', 'old preference', '2025-01-01')])
        new_scene(memory, date='2025-01-01')
        decide(memory, 'incoming', 'conflicts', targets=['old'])
        self.assertEqual(memory.graph.nodes['p']['content'], 'old preference')
        conflict = memory.graph.nodes['p']['persona_conflicts'][0]
        self.assertEqual(conflict['resolution'], 'unresolved')
        self.assertEqual(conflict['reason'], 'equal_source_time')

    def test_unknown_source_times_adopt_incoming_conclusion_and_persist_history(self):
        for old_date, new_date in [(None, '2025-01-01'), ('2025-01-01', None), (None, None)]:
            with self.subTest(old_date=old_date, new_date=new_date):
                memory = system()
                persona(memory, [claim('old', 'old preference', old_date),
                                 claim('unrelated', 'plays piano', '2030-01-01')])
                if new_date is None:
                    # A legacy scene can lack a recoverable source event time.
                    scene(memory, 'incoming', [])
                else:
                    new_scene(memory, date=new_date)
                decide(memory, 'incoming', 'conflicts', targets=['old'])
                self.assertEqual(memory.graph.nodes['p']['content'], 'plays piano\nnew preference')
                self.assertTrue(memory.graph.has_edge('p', 'incoming'))
                conflict = memory.graph.nodes['p']['persona_conflicts'][0]
                self.assertEqual(conflict['resolution'], 'replaced')
                self.assertEqual(conflict['reason'], 'unknown_source_time')
                self.assertEqual(conflict['previous_claims'][0]['content'], 'old preference')
                self.assertEqual(conflict['candidate']['evidence'][0]['scene_id'], 'incoming')
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'memory.pkl'
                    memory.save(path)
                    restored = system(storage_path=path)
                self.assertEqual(restored.graph.nodes['p'], memory.graph.nodes['p'])

    def test_unknown_time_in_targeted_claims_prefers_incoming_conclusion(self):
        memory = system()
        persona(memory, [claim('known', 'first preference', '2020-01-01'),
                         claim('unknown', 'second preference')])
        new_scene(memory)
        decide(memory, 'incoming', 'conflicts', targets=['known', 'unknown'])
        self.assertEqual(memory.graph.nodes['p']['persona_conflicts'][0]['resolution'], 'replaced')
        self.assertEqual(memory.graph.nodes['p']['content'], 'new preference')
        self.assertEqual(len(memory.graph.nodes['p']['persona_conflicts'][0]['previous_claims']), 2)

    def test_supporting_scene_adds_provenance_without_rewriting_claim(self):
        memory = system()
        persona(memory, [claim('music', 'likes piano', '2020-01-01')])
        new_scene(memory)
        decide(memory, 'incoming', 'supports', content='redundant paraphrase', targets=['music'])
        self.assertEqual(memory.graph.nodes['p']['content'], 'likes piano')
        self.assertEqual(len(memory.graph.nodes['p']['persona_claims'][0]['evidence']), 2)
        self.assertTrue(memory.graph.has_edge('p', 'incoming'))

    def test_unrelated_scene_leaves_profile_unchanged(self):
        memory = system()
        persona(memory, [claim('music', 'likes piano', '2020-01-01')])
        new_scene(memory)
        before = copy.deepcopy(memory.graph.nodes['p'])
        decide(memory, 'incoming', 'unrelated')
        self.assertEqual(memory.graph.nodes['p'], before)
        self.assertFalse(memory.graph.has_edge('p', 'incoming'))

    def test_processed_scene_is_idempotent(self):
        memory = system()
        new_scene(memory)
        decide(memory, 'incoming', 'adds', content='likes piano')
        with patch.object(memory.persona_manager, 'analyze_scene', side_effect=AssertionError('already processed')):
            memory._update_persona_incrementally(['incoming'])
        profiles = [d for _, d in memory.graph.nodes(data=True) if d.get('level') == 'persona']
        self.assertEqual(len(profiles), 1)
        self.assertEqual(len(profiles[0]['persona_claims']), 1)

    def test_deleted_scene_provenance_survives_compression_and_restart(self):
        memory = system()
        fact(memory, 'a', timestamp='2019-01-01')
        fact(memory, 'b', timestamp='2020-01-01')
        scene(memory, 'original', ['a', 'b'])
        memory.graph.add_node('p', level='persona', attribute_type='interests', content='old preference')
        memory.graph.add_edge('p', 'original', type='contains')
        memory._compress_scene('original')
        self.assertNotIn('original', memory.graph)
        evidence = memory.graph.nodes['p']['persona_claims'][0]['evidence'][0]
        self.assertEqual(evidence['scene_id'], 'original')
        self.assertEqual(evidence['timestamp_end'], '2020-01-01T00:00:00+00:00')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pkl'
            memory.save(path)
            restored = system(storage_path=path)
        target = restored.graph.nodes['p']['persona_claims'][0]['id']
        new_scene(restored, date='2021-01-01')
        decide(restored, 'incoming', 'conflicts', targets=[target])
        self.assertEqual(restored.graph.nodes['p']['content'], 'new preference')
        self.assertEqual(restored.graph.nodes['p']['persona_conflicts'][0]['resolution'], 'replaced')

    def test_updates_to_existing_scene_do_not_change_original_evidence_clock(self):
        memory = system()
        fact(memory, 'a', timestamp='2020-01-01')
        scene(memory, 'original', ['a'])
        memory.graph.add_node('p', level='persona', attribute_type='interests', content='old preference')
        memory.graph.add_edge('p', 'original', type='contains')
        memory.add_note('music later fact', time='2030-01-01')
        target = memory.graph.nodes['p']['persona_claims'][0]['id']
        new_scene(memory, date='2025-01-01')
        decide(memory, 'incoming', 'conflicts', targets=[target])
        self.assertEqual(memory.graph.nodes['p']['content'], 'new preference')
        self.assertEqual(memory.graph.nodes['original']['timestamp_end'], '2030-01-01T00:00:00+00:00')
        self.assertEqual(memory.graph.nodes['p']['persona_conflicts'][0]['previous_evidence_time'], '2020-01-01T00:00:00+00:00')

    def test_invalid_classification_rolls_back_input_and_new_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pkl'
            memory = system(storage_path=path, unmatched_threshold=2)
            memory.add_note('music first', time='2020-01-01')
            before = memory.graph.copy()
            memory.llm_controller.llm.fail_on = 'Analyze ONE newly created scene'
            with self.assertRaises(ValueError):
                memory.add_note('music second', time='2025-01-01')
            self.assertTrue(nx.utils.graphs_equal(memory.graph, before))
            self.assertTrue(nx.utils.graphs_equal(system(storage_path=path).graph, before))
            self.assertEqual(memory.turn_count, 1)

    def test_unknown_target_rejected_before_applying_any_proposal(self):
        memory = system()
        persona(memory, [claim('old', 'likes piano', '2020-01-01')])
        new_scene(memory)
        before = copy.deepcopy(memory.graph.nodes['p'])
        with self.assertRaises(ValueError):
            decide(memory, 'incoming', 'conflicts', targets=['invented-id'])
        self.assertEqual(memory.graph.nodes['p'], before)

    def test_old_ranking_config_is_migrated_and_ignored(self):
        memory = system()
        memory.graph.graph['incremental_config']['persona_recent_fraction'] = None
        self.assertNotIn('persona_recent_fraction', memory.incremental_config)
        new_scene(memory)
        decide(memory, 'incoming', 'adds', content='likes piano')
        self.assertTrue(any(d.get('level') == 'persona' for _, d in memory.graph.nodes(data=True)))


if __name__ == '__main__':
    unittest.main()
