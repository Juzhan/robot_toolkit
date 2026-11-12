"""JSON wire codec for the agent env RPC convention.

The envelope and numpy tags intentionally match the widely used RPent-style
``env.*`` protocol so an external agent can talk to this toolkit without
importing RPent itself.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Any

import numpy as np


def encode(value: Any) -> Any:
    """Recursively convert numpy values into JSON-safe tagged dictionaries."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return encode(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return {
            '__ndarray__': base64.b64encode(np.asarray(value).tobytes()).decode('ascii'),
            'dtype': str(value.dtype),
            'shape': list(value.shape),
        }
    if isinstance(value, np.generic):
        item = value.item()
        if value.dtype.kind in 'biuf' and isinstance(item, (bool, int, float)):
            return {'__npscalar__': item, 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value

def decode(value: Any) -> Any:
    """Recursively restore tagged numpy arrays and scalars."""
    if isinstance(value, dict):
        if '__ndarray__' in value and set(value) <= {'__ndarray__', 'dtype', 'shape'}:
            raw = base64.b64decode(value['__ndarray__'])
            array = np.frombuffer(raw, dtype=value.get('dtype'))
            return array.reshape(value.get('shape', (-1,))).copy()
        if '__npscalar__' in value and set(value) <= {'__npscalar__', 'dtype'}:
            return np.dtype(value['dtype']).type(value['__npscalar__'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def dumps(value: Any) -> str:
    return json.dumps(encode(value), separators=(',', ':'))


def loads(text: str | bytes) -> Any:
    return decode(json.loads(text))
