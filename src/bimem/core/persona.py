"""Scene-driven persona claims with durable evidence and temporal conflicts."""

import copy
import uuid
from datetime import datetime, timezone


DIMENSIONS = ('basic_info', 'interests', 'personality', 'values', 'relationships')


def parsed_time(value):
    # Avoid an import cycle with the graph maintenance mixin.
    from .incremental import event_time
    return event_time(value)


class PersonaMemoryMixin:
    def _initialize_persona_evidence(self):
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('level') == 'persona':
                self._ensure_persona_claims(node_id)

    def _scene_evidence(self, scene_id):
        self._set_scene_times(scene_id)
        attrs = self.graph.nodes[scene_id]
        # Generation/refinement time must never stand in for source event time.
        return {
            'scene_id': scene_id,
            'content': attrs.get('content', '').split('\n[Insight]:', 1)[0],
            'timestamp_start': attrs.get('timestamp_start'),
            'timestamp_end': attrs.get('timestamp_end'),
        }

    def _ensure_persona_claims(self, persona_id):
        """Migrate legacy text conservatively, without inventing source times."""
        attrs = self.graph.nodes[persona_id]
        if 'persona_claims' not in attrs:
            evidence = [self._scene_evidence(n) for n in self.graph.successors(persona_id)
                        if self.graph.nodes[n].get('level') == 'scene'
                        and self.graph.edges[persona_id, n].get('type') == 'contains']
            text = attrs.get('content', '').strip()
            attrs['persona_claims'] = ([{
                'id': str(uuid.uuid4()), 'content': text,
                'evidence': evidence, 'legacy': True,
            }] if text else [])
        return attrs['persona_claims']

    @staticmethod
    def _evidence_time(claims):
        if not claims or any(not claim.get('evidence') for claim in claims):
            return None
        evidence = [source for claim in claims for source in claim.get('evidence', [])]
        if not evidence or any(source.get('timestamp_end') is None for source in evidence):
            return None
        return max(parsed_time(source['timestamp_end']) for source in evidence)

    def _refresh_persona(self, persona_id):
        attrs = self.graph.nodes[persona_id]
        claims = attrs['persona_claims']
        attrs['content'] = '\n'.join(claim['content'] for claim in claims)
        attrs['source_scene_ids'] = list(dict.fromkeys(
            source['scene_id'] for claim in claims for source in claim.get('evidence', [])
        ))
        known_times = [parsed_time(source['timestamp_end']) for claim in claims
                       for source in claim.get('evidence', []) if source.get('timestamp_end')]
        attrs['evidence_timestamp'] = max(known_times).isoformat() if known_times else None
        # Keep a separate system update time; conflict decisions use claim evidence.
        attrs['updated_at'] = datetime.now(timezone.utc).isoformat()
        active_scenes = set(attrs['source_scene_ids'])
        for target in list(self.graph.successors(persona_id)):
            if (self.graph.nodes[target].get('level') == 'scene'
                    and self.graph.edges[persona_id, target].get('type') == 'contains'
                    and target not in active_scenes):
                self.graph.remove_edge(persona_id, target)
        for scene_id in active_scenes:
            if scene_id in self.graph and self.graph.nodes[scene_id].get('level') == 'scene':
                self.graph.add_edge(persona_id, scene_id, type='contains')

    def _apply_persona_decision(self, scene_id, decision, existing):
        dimension = decision['dimension']
        persona_id = existing.get(dimension)
        if persona_id is None:
            persona_id = str(uuid.uuid4())
            self.graph.add_node(persona_id, level='persona', attribute_type=dimension,
                                content='', tags=[dimension.replace('_', ' ').title()],
                                timestamp=datetime.now(timezone.utc).isoformat(), persona_claims=[])
            existing[dimension] = persona_id
        attrs = self.graph.nodes[persona_id]
        claims = self._ensure_persona_claims(persona_id)
        targets = [c for c in claims if c['id'] in decision['target_claim_ids']]
        evidence = self._scene_evidence(scene_id)
        candidate = {'id': str(uuid.uuid4()), 'content': decision['content'],
                     'evidence': [evidence]}
        relation = decision['relation']
        if relation == 'supports':
            # Confirmation adds provenance without rewriting an existing conclusion.
            for claim in targets:
                if not any(source == evidence for source in claim['evidence']):
                    claim['evidence'].append(copy.deepcopy(evidence))
        elif relation == 'adds':
            claims.append(candidate)
        else:
            old_time = self._evidence_time(targets)
            new_time = parsed_time(evidence['timestamp_end']) if evidence['timestamp_end'] else None
            if old_time is None or new_time is None:
                resolution, reason = 'replaced', 'unknown_source_time'
            elif new_time == old_time:
                resolution, reason = 'unresolved', 'equal_source_time'
            elif new_time > old_time:
                resolution, reason = 'replaced', 'newer_scene_evidence'
            else:
                resolution, reason = 'retained', 'older_scene_evidence'
            attrs.setdefault('persona_conflicts', []).append({
                'id': str(uuid.uuid4()), 'candidate': copy.deepcopy(candidate),
                'previous_claims': copy.deepcopy(targets),
                'previous_evidence_time': old_time.isoformat() if old_time else None,
                'incoming_evidence_time': new_time.isoformat() if new_time else None,
                'resolution': resolution, 'reason': reason,
            })
            if resolution == 'replaced':
                targeted = set(decision['target_claim_ids'])
                claims[:] = [c for c in claims if c['id'] not in targeted] + [candidate]
        self._refresh_persona(persona_id)

    def _update_persona_incrementally(self, new_scenes):
        for scene_id in dict.fromkeys(new_scenes):
            if (scene_id not in self.graph or self.graph.nodes[scene_id].get('level') != 'scene'
                    or self.graph.nodes[scene_id].get('persona_processed')):
                continue
            profiles = {}
            existing = {}
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get('level') == 'persona' and attrs.get('attribute_type') in DIMENSIONS:
                    dimension = attrs['attribute_type']
                    if dimension in existing:
                        raise ValueError(f'Multiple persona nodes for dimension {dimension}')
                    existing[dimension] = node_id
                    profiles[dimension] = copy.deepcopy(self._ensure_persona_claims(node_id))
            # The manager returns only classified proposals. Core code controls
            # evidence, time comparison, replacement and audit records.
            decisions = self.persona_manager.analyze_scene(self._scene_evidence(scene_id), profiles)
            for decision in decisions:
                self._apply_persona_decision(scene_id, decision, existing)
            self.graph.nodes[scene_id]['persona_processed'] = True
            self.graph.nodes[scene_id]['persona_update_pending'] = False
