import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from offline_memory import system
from src.bimem.core.incremental import MemoryCapacityError, event_time


def fact(memory, node_id, content='music fact', timestamp='2020-01-01'):
    memory.graph.add_node(node_id, content=content, level='fact', timestamp=timestamp,
                          context='', keywords=[], tags=[])


def scene(memory, node_id, facts, content='music scene', timestamp='2099-01-01'):
    memory.graph.add_node(node_id, content=content, level='scene', timestamp=timestamp,
                          keywords=[], tags=[])
    for fact_id in facts:
        memory.graph.add_edge(node_id, fact_id, type='contains')


class IncrementalTests(unittest.TestCase):
    def test_best_match_updates_only_one_scene(self):
        memory = system()
        fact(memory, 'old_music')
        fact(memory, 'old_sport', 'sport fact')
        scene(memory, 'music_scene', ['old_music'])
        scene(memory, 'sport_scene', ['old_sport'], 'sport scene')
        new_id = memory.add_note('music piano lesson', time='2026-10-24')
        self.assertTrue(memory.graph.has_edge('music_scene', new_id))
        self.assertFalse(memory.graph.has_edge('sport_scene', new_id))
        self.assertNotIn(new_id, memory.unmatched_node_ids)
        self.assertIn('music piano lesson', memory.graph.nodes['music_scene']['content'])
        self.assertEqual(memory.graph.nodes['sport_scene']['content'], 'sport scene')
        self.assertEqual(memory.graph.nodes['music_scene']['timestamp_end'], '2026-10-24T00:00:00+00:00')
        update_prompt = next(p for p in memory.llm_controller.llm.prompts if p.startswith('Incrementally update a scene'))
        self.assertNotIn('music fact', update_prompt)  # Prior facts are not re-sent.
        self.assertEqual(memory.turn_count, 1)
        self.assertEqual(len(memory.retriever.corpus), len(memory.graph))

    def test_unmatched_99_waits_100_clusters_without_touching_existing_scene(self):
        memory = system()
        fact(memory, 'known', 'sport fact')
        scene(memory, 'known_scene', ['known'], 'sport scene')
        for i in range(98):
            fact(memory, f'pool{i}')
        memory.add_note('music note 99', time='2026-01-01')
        self.assertEqual(len(memory.unmatched_node_ids), 99)
        self.assertEqual(len([n for n, d in memory.graph.nodes(data=True) if d['level'] == 'scene']), 1)
        memory.add_note('music note 100', time='2026-01-02')
        self.assertEqual(memory.unmatched_node_ids, [])
        self.assertEqual(memory._scene_facts('known_scene'), ['known'])
        self.assertEqual(memory.graph.nodes['known_scene']['content'], 'sport scene')
        scenes = [n for n, d in memory.graph.nodes(data=True) if d['level'] == 'scene' and n != 'known_scene']
        self.assertEqual(len(scenes), 1)
        self.assertEqual(len(memory._scene_facts(scenes[0])), 100)

    def test_1001_compresses_scene_with_oldest_latest_fact(self):
        memory = system()
        fact(memory, 'a_old', 'sport fact', '1900-01-01')
        fact(memory, 'a_new', 'sport fact', '2026-01-01')
        fact(memory, 'b_old', 'music fact', '2019-01-01')
        fact(memory, 'b_new', 'music fact', '2020-01-01')
        scene(memory, 'a', ['a_old', 'a_new'], 'sport scene', timestamp='1900-01-01')
        scene(memory, 'b', ['b_old', 'b_new'], 'music scene', timestamp='2099-01-01')
        for i in range(996):
            fact(memory, f'c{i}', 'travel fact', '2025-01-01')
        scene(memory, 'c', [f'c{i}' for i in range(996)], 'travel scene')
        memory.graph.add_edge('b_old', 'c0', type='fact_related', weight=.9)
        memory.graph.add_node('persona', level='persona', attribute_type='interests', content='music')
        memory.graph.add_edge('persona', 'b', type='contains')
        new_id = memory.add_note('sport new event', time='2026-10-24')
        self.assertEqual(len(memory.fact_node_ids), 1000)
        self.assertIn(new_id, memory.graph)
        self.assertIn('a', memory.graph)
        self.assertIn('c', memory.graph)
        self.assertNotIn('b', memory.graph)
        self.assertNotIn('b_old', memory.graph)
        self.assertNotIn('b_new', memory.graph)
        compressed = memory.unmatched_node_ids
        self.assertEqual(len(compressed), 1)
        attrs = memory.graph.nodes[compressed[0]]
        self.assertEqual(attrs['timestamp_start'], '2019-01-01T00:00:00+00:00')
        self.assertEqual(attrs['timestamp_end'], '2020-01-01T00:00:00+00:00')
        self.assertEqual(attrs['source_fact_count'], 2)
        self.assertTrue(memory.graph.has_edge(compressed[0], 'c0'))
        self.assertFalse(memory.graph.has_edge('persona', compressed[0]))

    def test_compressed_fact_enters_pool_before_clustering(self):
        memory = system(max_fact_nodes=3, unmatched_threshold=3)
        fact(memory, 'old1')
        fact(memory, 'old2')
        scene(memory, 'old_scene', ['old1', 'old2'])
        fact(memory, 'pending', 'travel fact')
        order = []
        original_match = memory._match_scene
        original_compress = memory._enforce_fact_capacity
        original_cluster = memory._cluster_unmatched

        def match(*args):
            order.append('match')
            return original_match(*args)

        def compress(*args):
            order.append('compress')
            return original_compress(*args)

        def cluster(*args):
            order.append('cluster')
            self.assertEqual(len(memory.unmatched_node_ids), 3)
            self.assertTrue(any(memory.graph.nodes[n].get('compressed') for n in memory.unmatched_node_ids))
            return original_cluster(*args)

        with patch.object(memory, '_match_scene', side_effect=match), patch.object(memory, '_enforce_fact_capacity', side_effect=compress), patch.object(memory, '_cluster_unmatched', side_effect=cluster):
            memory.add_note('travel new', time='2026-01-01')
        self.assertEqual(order, ['match', 'compress', 'cluster'])
        self.assertEqual(len(memory.fact_node_ids), 3)
        self.assertEqual(memory.unmatched_node_ids, [])
        self.assertNotIn('old_scene', memory.graph)

    def test_capacity_without_compressible_scene_rolls_back(self):
        memory = system(max_fact_nodes=1)
        fact(memory, 'original')
        memory._update_retriever([])
        before = memory.graph.copy()
        with self.assertRaises(MemoryCapacityError):
            memory.add_note('travel unmatched', time='2026-01-01')
        self.assertTrue(nx.utils.graphs_equal(before, memory.graph))
        self.assertEqual(memory.turn_count, 0)
        self.assertEqual(len(memory.retriever.corpus), 1)

    def test_failed_compression_preserves_original_scene_and_facts(self):
        memory = system(max_fact_nodes=2)
        fact(memory, 'a')
        fact(memory, 'b')
        scene(memory, 's', ['a', 'b'])
        before = memory.graph.copy()
        memory.llm_controller.llm.fail_on = 'Compress the following'
        with self.assertRaises(ValueError):
            memory.add_note('travel', time='2026-01-01')
        self.assertTrue(nx.utils.graphs_equal(before, memory.graph))

    def test_automatic_restore_pool_and_config_then_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pkl'
            memory = system(storage_path=path, unmatched_threshold=2)
            memory.add_note('music first', time='2026-01-01')
            restored = system(storage_path=path)
            self.assertEqual(len(restored.unmatched_node_ids), 1)
            self.assertEqual(restored.incremental_config['unmatched_threshold'], 2)
            restored.add_note('music second', time='2026-01-02')
            self.assertEqual(restored.unmatched_node_ids, [])
            self.assertEqual(system(storage_path=path).turn_count, 2)

    def test_failed_disk_save_rolls_back_in_memory_and_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pkl'
            memory = system(storage_path=path)
            memory.add_note('music original', time='2026-01-01')
            before = memory.graph.copy()
            with patch('src.bimem.core.persistence.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    memory.add_note('music second', time='2026-01-02')
            self.assertTrue(nx.utils.graphs_equal(before, memory.graph))
            self.assertTrue(nx.utils.graphs_equal(before, system(storage_path=path).graph))
            self.assertEqual(memory.turn_count, 1)

    def test_existing_scene_update_does_not_trigger_persona(self):
        memory = system()
        fact(memory, 'old')
        scene(memory, 's', ['old'])
        memory.graph.add_node('interests', level='persona', attribute_type='interests', content='previous')
        memory.add_note('music newest', time='2026-02-01')
        self.assertEqual(memory.graph.nodes['interests']['content'], 'previous')
        self.assertFalse(any(p.startswith('Analyze ONE newly created scene') for p in memory.llm_controller.llm.prompts))

    def test_event_times_support_both_datasets_and_timezones(self):
        self.assertEqual(event_time('1:56 pm on 8 May, 2023'), event_time('2023-05-08T13:56:00Z'))
        self.assertEqual(event_time('2023/05/20 (Sat) 02:21'), event_time('2023-05-20T10:21:00+08:00'))
        with self.assertRaises(ValueError):
            event_time('not a date')


if __name__ == '__main__':
    unittest.main()
