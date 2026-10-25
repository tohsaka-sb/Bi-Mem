"""Incremental graph maintenance, independent of model/backend construction."""

import uuid
from datetime import datetime, timezone

import networkx as nx
import numpy as np

from .persona import PersonaMemoryMixin


class MemoryCapacityError(RuntimeError):
    """No scene with multiple facts can be compressed under the current policy."""


def event_time(value):
    """Parse ISO, LoCoMo and LongMemEval event times into comparable UTC times."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError:
            for pattern in ('%I:%M %p on %d %B, %Y', '%Y/%m/%d (%a) %H:%M'):
                try:
                    parsed = datetime.strptime(text, pattern)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f'Unsupported memory timestamp: {value!r}')
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


DEFAULT_CONFIG = {
    'scene_match_threshold': 0.5,
    'fact_link_threshold': 0.2,
    'unmatched_threshold': 100,
    'max_fact_nodes': 1000,
}


def validate_config(config):
    result = dict(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        raise ValueError('Invalid incremental memory configuration')
    config = dict(config)
    # Migrate snapshots from the provisional ranking policy; it no longer gates updates.
    config.pop('persona_recent_fraction', None)
    if set(config) - set(result):
        raise ValueError('Invalid incremental memory configuration')
    result.update(config)
    for name in ('scene_match_threshold', 'fact_link_threshold'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1 <= value <= 1:
            raise ValueError(f'{name} must be between -1 and 1')
    for name in ('unmatched_threshold', 'max_fact_nodes'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    return result


class IncrementalMemoryMixin(PersonaMemoryMixin):
    @property
    def incremental_config(self):
        # Graph metadata survives both core snapshots and legacy graph-only saves.
        return validate_config(self.graph.graph.get('incremental_config', {}))

    @property
    def fact_node_ids(self):
        return [n for n, attrs in self.graph.nodes(data=True) if attrs.get('level') == 'fact']

    def _scene_facts(self, scene_id):
        return [n for n in self.graph.successors(scene_id)
                if self.graph.nodes[n].get('level') == 'fact'
                and self.graph.edges[scene_id, n].get('type') == 'contains']

    @property
    def unmatched_node_ids(self):
        """The pool is derived from contains edges, so restart cannot lose it."""
        return [n for n in self.fact_node_ids if not any(
            self.graph.nodes[parent].get('level') == 'scene'
            and self.graph.edges[parent, n].get('type') == 'contains'
            for parent in self.graph.predecessors(n)
        )]

    def _content_scores(self, content, node_ids):
        if not node_ids:
            return np.empty(0)
        cache = getattr(self, '_content_embeddings', {})
        texts = [content] + [self.graph.nodes[n].get('content', '') for n in node_ids]
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            cache.update(zip(missing, self.retriever.model.encode(missing)))
        self._content_embeddings = cache
        query = np.asarray(cache[content])
        matrix = np.asarray([cache[t] for t in texts[1:]])
        divisor = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        return np.divide(matrix @ query, divisor, out=np.zeros(len(matrix)), where=divisor != 0)

    def _fact_end(self, node_id):
        attrs = self.graph.nodes[node_id]
        return event_time(attrs.get('timestamp_end', attrs.get('timestamp')))

    def _set_scene_times(self, scene_id):
        facts = self._scene_facts(scene_id)
        if not facts:
            return
        starts = [event_time(self.graph.nodes[n].get('timestamp_start', self.graph.nodes[n]['timestamp']))
                  for n in facts]
        self.graph.nodes[scene_id]['timestamp_start'] = min(starts).isoformat()
        self.graph.nodes[scene_id]['timestamp_end'] = max(self._fact_end(n) for n in facts).isoformat()

    def _match_scene(self, fact_id):
        scenes = [n for n, attrs in self.graph.nodes(data=True) if attrs.get('level') == 'scene']
        scores = self._content_scores(self.graph.nodes[fact_id]['content'], scenes)
        if not len(scores):
            return None
        index = int(np.argmax(scores))
        score = float(scores[index])
        if score < self.incremental_config['scene_match_threshold']:
            return None
        scene_id = scenes[index]
        changes = self.scene_manager.update_scene(self.graph.nodes[scene_id], self.graph.nodes[fact_id])
        self.graph.add_edge(scene_id, fact_id, type='contains', weight=score)
        self.graph.nodes[scene_id].update(changes)
        self.graph.nodes[scene_id]['refined'] = False
        self._set_scene_times(scene_id)
        return scene_id

    def _link_fact(self, fact_id):
        others = [n for n in self.fact_node_ids if n != fact_id]
        scores = self._content_scores(self.graph.nodes[fact_id]['content'], others)
        for other, score in zip(others, scores):
            if score > self.incremental_config['fact_link_threshold']:
                self.graph.add_edge(fact_id, other, type='fact_related', weight=float(score))
                self.graph.add_edge(other, fact_id, type='fact_related', weight=float(score))

    def _compress_scene(self, scene_id):
        self._initialize_persona_evidence()
        facts = self._scene_facts(scene_id)
        if len(facts) < 2:
            raise ValueError('Compression requires at least two facts')
        summary = self.scene_manager.compress_facts(self.graph, facts)
        starts = [event_time(self.graph.nodes[n].get('timestamp_start', self.graph.nodes[n]['timestamp']))
                  for n in facts]
        start = min(starts).isoformat()
        end = max(self._fact_end(n) for n in facts).isoformat()
        compressed_id = str(uuid.uuid4())
        self.graph.add_node(
            compressed_id, level='fact', timestamp=start, timestamp_start=start,
            timestamp_end=end, compressed=True,
            source_fact_count=sum(self.graph.nodes[n].get('source_fact_count', 1) for n in facts),
            **summary,
        )
        # Preserve external fact associations, but no scene parent: the summary
        # explicitly re-enters the unmatched pool, including for legacy overlaps.
        removed = set(facts) | {scene_id}
        other_parents = {parent for fact in facts for parent in self.graph.predecessors(fact)
                         if parent not in removed and self.graph.nodes[parent].get('level') == 'scene'}
        for fact in facts:
            for left, right, attrs in list(self.graph.in_edges(fact, data=True)) + list(self.graph.out_edges(fact, data=True)):
                external = left if right == fact else right
                if external in removed or self.graph.nodes[external].get('level') != 'fact':
                    continue
                edge = (external, compressed_id) if right == fact else (compressed_id, external)
                self.graph.add_edge(*edge, **dict(attrs))
        self.graph.remove_nodes_from(removed)
        # Other parents can exist in older checkpoints. Refresh their coverage.
        for node_id in other_parents:
            remaining = self._scene_facts(node_id)
            if not remaining:
                self.graph.remove_node(node_id)
            else:
                self._set_scene_times(node_id)
        return compressed_id

    def _enforce_fact_capacity(self):
        replacements = {}
        while len(self.fact_node_ids) > self.incremental_config['max_fact_nodes']:
            candidates = [n for n, attrs in self.graph.nodes(data=True)
                          if attrs.get('level') == 'scene' and len(self._scene_facts(n)) >= 2]
            if not candidates:
                raise MemoryCapacityError(
                    'Fact limit reached, but no scene contains multiple facts. '
                    'No unapproved fallback merge was performed; the input was rolled back.'
                )
            # Minimize each scene's latest fact time, never its generation time.
            oldest = min(candidates, key=lambda n: max(self._fact_end(f) for f in self._scene_facts(n)))
            old_facts = self._scene_facts(oldest)
            compressed_id = self._compress_scene(oldest)
            replacements.update(dict.fromkeys(old_facts, compressed_id))
        return replacements

    def _cluster_unmatched(self, force=False, save_callback=None):
        pending = [n for n, attrs in self.graph.nodes(data=True)
                   if attrs.get('level') == 'scene' and attrs.get('persona_update_pending')]
        if pending:
            self._update_persona_incrementally(pending)
        pool = self.unmatched_node_ids
        if not pool or (not force and len(pool) < self.incremental_config['unmatched_threshold']):
            return []
        subgraph = self.graph.subgraph(pool).to_undirected()
        communities = nx.algorithms.community.label_propagation_communities(subgraph)
        new_scenes = []
        for community in communities:
            # Keep graph insertion order for stable prompts and edge order.
            facts = [n for n in pool if n in community]
            data = dict(self.scene_manager.build_scene(self.graph, facts))
            scene_id = data.pop('id')
            data['persona_update_pending'] = True
            self.graph.add_node(scene_id, **data)
            for fact in facts:
                self.graph.add_edge(scene_id, fact, type='contains')
            self._set_scene_times(scene_id)
            self._update_persona_incrementally([scene_id])
            new_scenes.append(scene_id)
            if save_callback:
                self._update_retriever([scene_id])
                save_callback()
        return new_scenes
