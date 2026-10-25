"""Load core classes without optional model runtimes, using a deterministic backend.

The compiled class definitions are the repository implementation, including the
constructor and complete add/load paths. Only embedding and LLM I/O are fake.
"""
import ast
import copy
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.bimem.core.incremental import IncrementalMemoryMixin, event_time, validate_config
from src.bimem.core.persistence import read_checkpoint, write_checkpoint


class Encoder:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.encode([texts])[0]
        self.calls.append(list(texts))
        return np.array([[t.lower().count('music') + t.lower().count('piano'),
                          t.lower().count('sport'), t.lower().count('travel'), .01]
                         for t in texts], dtype=float)


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.model = Encoder()
        self.corpus = []
        self.embeddings = None
        self.bm25 = None
        self.is_bm25_dirty = True


class FakeLLM:
    def __init__(self):
        self.prompts = []
        self.fail_on = None

    def get_completion(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            return '{}'
        if prompt.startswith('Incrementally update a scene'):
            old, new = prompt.split('Existing scene:\n', 1)[1].split('\nNew fact:\n', 1)
            return json.dumps({'scene_summary': json.loads(old)['content'] + '; ' + json.loads(new)['content'],
                               'keywords': ['updated'], 'tags': ['updated']})
        if prompt.startswith('Compress the following'):
            return json.dumps({'content': 'compressed music', 'context': 'retained core', 'keywords': [], 'tags': []})
        if prompt.startswith('Analyze ONE newly created scene'):
            return json.dumps({'decisions': [{'dimension': 'interests', 'relation': 'adds',
                                              'content': 'updated interests', 'target_claim_ids': []}]})
        if 'AI Scene Analyst' in prompt:
            return json.dumps({'scene_summary': 'music scene', 'keywords': [], 'tags': []})
        return json.dumps({'keywords': [], 'context': '', 'tags': []})


class FakeController:
    def __init__(self, *args, **kwargs):
        self.llm = FakeLLM()


def classes(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, 'exec'), namespace)


namespace = dict(nx=nx, np=np, Path=Path, read_checkpoint=read_checkpoint,
                 write_checkpoint=write_checkpoint, IncrementalMemoryMixin=IncrementalMemoryMixin,
                 event_time=event_time, validate_config=validate_config, copy=copy,
                 uuid=uuid, datetime=datetime, json=json, re=re,
                 HybridRetriever=FakeRetriever, LLMController=FakeController)
classes('src/bimem/core/memory_managers.py', {'BaseManager', 'SceneManager', 'PersonaManager'}, namespace)
classes('src/bimem/core/memory_layer.py', {'MemoryNote', 'AgenticMemorySystem'}, namespace)
MemorySystem = namespace['AgenticMemorySystem']


def system(**kwargs):
    return MemorySystem(**kwargs)
