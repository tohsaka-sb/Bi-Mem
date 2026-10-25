"""Offline persistence/index tests; no model downloads or API calls required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from src.bimem.core.persistence import read_checkpoint, write_checkpoint
from offline_memory import system


class PersistenceTests(unittest.TestCase):
    def test_roundtrip_and_continue_indexing(self):
        memory = system()
        memory.graph.add_node('fact', level='fact', content='first', timestamp='2026-01-01')
        memory.graph.add_node('scene', level='scene', content='theme')
        memory.graph.add_edge('scene', 'fact', type='contains')
        memory.graph.graph['pending'] = ['fact']
        memory.turn_count = 12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'state.pkl'
            memory.save(path)
            restored = system().load(path)
        self.assertTrue(nx.utils.graphs_equal(memory.graph, restored.graph))
        self.assertEqual(restored.turn_count, 12)
        restored.graph.add_node('new', level='fact', content='second')
        restored._update_retriever(['new'])
        self.assertEqual(len(restored.retriever.corpus), 3)

    def test_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pkl'
            write_checkpoint(path, {'graph': nx.DiGraph(), 'turns_processed': 17})
            self.assertEqual(system().load(path).turn_count, 17)

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pkl'
            write_checkpoint(path, {'value': 'original'})
            with patch('src.bimem.core.persistence.pickle.dump', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    write_checkpoint(path, {'value': 'changed'})
            self.assertEqual(read_checkpoint(path), {'value': 'original'})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_updates_and_deletions_keep_row_order(self):
        memory = system()
        memory.graph.add_node('a', content='old')
        memory.graph.add_node('b', content='other')
        memory._update_retriever(['a', 'b'])
        memory.graph.nodes['a']['content'] = 'updated'
        memory._update_retriever(['a'])
        self.assertEqual(len(memory.retriever.corpus), 2)
        self.assertIn('updated', memory.retriever.corpus[0])
        self.assertEqual(len(memory.retriever.model.calls[-1]), 1)
        memory.graph.remove_node('a')
        memory._update_retriever([])
        self.assertEqual(len(memory.retriever.corpus), 1)
        self.assertIn('other', memory.retriever.corpus[0])
        memory.graph.clear()
        memory._update_retriever([])
        self.assertEqual(memory.retriever.corpus, [])
        self.assertIsNone(memory.retriever.embeddings)

    def test_failed_load_keeps_existing_state(self):
        memory = system()
        memory.graph.add_node('original', content='original')
        memory._update_retriever(['original'])
        graph = nx.DiGraph()
        graph.add_node('replacement', content='replacement')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pkl'
            write_checkpoint(path, {'graph': graph, 'turn_count': 99})
            with patch.object(memory.retriever.model, 'encode', side_effect=RuntimeError('encoder failed')):
                with self.assertRaises(RuntimeError):
                    memory.load(path)
        self.assertEqual(list(memory.graph), ['original'])
        self.assertEqual(memory.turn_count, 0)
        self.assertIn('original', memory.retriever.corpus[0])

    def test_invalid_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pkl'
            write_checkpoint(path, {'format_version': 99})
            with self.assertRaises(ValueError):
                system().load(path)


if __name__ == '__main__':
    unittest.main()
