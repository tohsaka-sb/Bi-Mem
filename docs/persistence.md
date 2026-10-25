# Persistent incremental memory

```python
from src.bimem.core.memory_layer import AgenticMemorySystem

memory = AgenticMemorySystem(
    storage_path="output/user_memory.pkl",
    scene_match_threshold=0.5,
)
fact_id = memory.add_note("I started learning piano.", time="2026-10-24 09:00:00")
```

If the file exists, initialization restores it. Each completed `add_note()` saves
one atomic snapshot. Explicit saving/loading is also supported:

```python
memory.save("output/user_memory.pkl")
memory.load("output/user_memory.pkl")
```

## Input processing

1. Create one fact per input, preserving the original content and extracting
   keywords, context and tags using the LLM.
2. Match against scene contents by cosine similarity. Attach to the highest
   scoring scene if its score reaches `scene_match_threshold`. Update that scene
   from its previous summary and the new fact; avoid re-summarizing all facts.
   Facts without a matching scene remain in the unmatched pool.
3. Build semantic fact-to-fact associations. If there are more than 1000 facts,
   compress whole scenes until the count is within the limit. Rank scenes by
   their **latest constituent fact timestamp**, oldest first. Only scenes with
   at least two facts can reduce the count. Compress all their facts into one
   fact, remove the source facts and scene, and put the compressed fact in the
   unmatched pool. Preserve its time range, source fact count and external fact
   associations. A compressed fact uses its original end time for recency; the
   compression execution time does not make old evidence new.
4. When the unmatched pool reaches 100 facts, run LPA on its induced fact graph,
   then summarize each community into a new scene. Existing scene memberships
   are preserved. Isolated facts produce singleton scenes. Compressed facts
   participate in this step; they are not immediately re-matched to old scenes.
5. Each newly created scene is checked for relevance to the five persona
   dimensions, then for support, compatible additions or conflicts with specific
   existing claims. Compatible evidence is added regardless of its age. Conflict
   precedence is decided by code using source event times, not by the LLM.
   Merely updating an existing scene does not trigger persona updates.
6. Synchronize the index and save. Any failed maintenance, encoding or save
   operation rolls back this input's graph, index and input counter.

The returned fact ID remains live, including when the incoming fact itself was
part of a compressed scene. The unmatched pool is derived from scene `contains`
edges, so there is no separate pool file that can drift out of sync.

## Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `scene_match_threshold` | `0.5` | Minimum cosine similarity to the best scene; configurable starting value, not calibrated on this repository's data |
| `fact_link_threshold` | `0.2` | Link facts whose cosine similarity exceeds this value |
| `unmatched_threshold` | `100` | Pool size that triggers incremental clustering |
| `max_fact_nodes` | `1000` | Maximum number of accepted fact nodes |

## Persona evidence and conflicts

Persona nodes retain their five dimensions, but each stores `persona_claims`
with stable claim IDs and supporting scene snapshots. A snapshot records the
scene ID, summary and original fact event time range. It survives deletion or
compression of the source scene. The profile text is composed from accepted
claims; only targeted conflicting claims can be replaced.

The LLM classifies each new scene as unrelated, adding compatible information,
supporting known claims, or conflicting with specified claim IDs. It receives
scene summaries and existing persona claims; raw facts do not directly update
persona. Code validates the classification and applies the decision:

| Relation | Result |
| --- | --- |
| Unrelated | Leave the persona unchanged |
| Compatible addition | Add the conclusion and its evidence, regardless of event age |
| Support | Retain the conclusion and append supporting evidence |
| Conflict, incoming evidence newer | Replace only the contradictory claims |
| Conflict, existing evidence newer | Retain existing claims |
| Conflict, unknown source times | Adopt the incoming scene's conclusion and archive the replaced claims and evidence |
| Conflict, equal source times | Retain existing claims and record an unresolved conflict |

Conflict comparison uses the new scene's latest constituent fact event time
and the latest supporting source time of the **targeted claims**. It does not
use scene generation time, persona modification time, or timestamps supporting
unrelated claims in the same dimension. If either side lacks source times,
the incoming scene's conclusion takes precedence. Equal known times remain
unresolved.

`persona_conflicts` stores previous claims and evidence, the incoming candidate,
compared event times and resolution. Accepted evidence is frozen: extending a
source scene later cannot silently change the provenance time of an already
accepted persona conclusion. Unchanged claims and dimension node IDs are kept.

Old persona text is migrated to one legacy claim per dimension, using available
scene `contains` edges for evidence. If those source links no longer exist, its
source time is unknown; the system does not substitute its generation timestamp.
Legacy text cannot provide precise per-sentence provenance retroactively. New
claims have their own source records. The provisional `persona_recent_fraction`
key in older snapshots is ignored; ranking no longer gates persona updates.

Each created scene has `persona_processed`/`persona_update_pending` metadata, so
retries do not apply it twice. Explicit batch processing resumes pending updates
before clustering further facts. New scenes update persona once during creation;
existing-scene summary updates do not automatically re-submit them.

If capacity is exceeded and no scene contains multiple facts, `add_note()` raises
`MemoryCapacityError` and rolls back the input. No independent facts are merged
without a confirmed fallback policy. The caller still holds the input for retry.

Configuration lives in graph metadata and is restored from checkpoints,
overriding constructor values for an existing snapshot. For an intentional
policy change, update the metadata with validated values and save:

```python
from src.bimem.core.incremental import validate_config

config = memory.incremental_config
config["scene_match_threshold"] = 0.6
memory.graph.graph["incremental_config"] = validate_config(config)
memory.save("output/user_memory.pkl")
```

## Recovery and compatibility

Snapshots contain graph, input count, evolution counters and configuration.
The pool, compressed fact metadata, scene time ranges, persona content and all
edges are preserved as graph state. Loading accepts old evaluation checkpoints
containing `graph` and `turns_processed`. Their pool is reconstructed from actual
scene membership. Old snapshots without configuration use the defaults above.

The retrieval index is derived from graph content using the configured embedding
model. Loading calls no LLM. Model/backend connections and credentials remain
runtime configuration. ISO timestamps, LoCoMo timestamps and LongMemEval
timestamps are supported; timestamps without timezone offsets share a common
UTC comparison basis. Invalid timestamps are rejected when used for processing.

`test_locomo.py` and `test_longmem.py` use core snapshots and enable saving after
each input. Separate retriever cache files remain available for compatibility,
but recovery rebuilds from graph content rather than relying on those caches.

Explicit `build_hierarchy_batch()` flushes even a pool smaller than 100 and can
build the initial persona and apply reflective refinement. It only clusters
unmatched facts. For batch operations or direct graph mutations, save explicitly
or pass a batch `save_callback`. A legacy graph already above the fact cap is
loaded intact; the next successful input enforces the cap.

Saving flushes a temporary file in the destination directory before replacing
the previous snapshot. Failed writes leave the previous file intact. Load only
trusted local pickle files. Concurrent writers to the same snapshot are not
supported.

## Offline verification

Tests use NumPy, NetworkX and the standard library. Actual core and manager class
bodies run against fake LLM and embedding I/O without model downloads or API
calls. Full inference dependencies are still required to run the application.

```bash
python -m unittest discover -s tests -v
```
