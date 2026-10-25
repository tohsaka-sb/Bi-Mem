"""Atomic local checkpoints. Only load pickle files from trusted sources."""

import os
import pickle
import tempfile
from pathlib import Path


def write_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=path.parent, prefix=f'.{path.name}.', delete=False
        ) as stream:
            temporary = stream.name
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    with Path(path).open('rb') as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a dictionary')
    version = state.get('format_version', 1)
    if version != 1:
        raise ValueError(f'Unsupported checkpoint version: {version}')
    return state
