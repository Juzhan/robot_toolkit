"""JSON-backed configuration with dataclass defaults.

The same pattern as the existing projects (a JSON file on disk, dataclasses in
code) but recursive, so nested robot / gripper / camera sections work without
per-section boilerplate.
"""

from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from pathlib import Path


def load_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataclasses.is_dataclass(data):
        data = dataclasses.asdict(data)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path


def _resolve_hints(cls) -> dict:
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {}


def _is_optional(annotation) -> tuple[bool, object]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        arguments = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(arguments) == 1:
            return True, arguments[0]
    return False, annotation


def build_dataclass(cls, data):
    """Instantiate ``cls`` from a dict, coercing nested dataclasses.

    Unknown keys are ignored on purpose: config files are forward compatible
    and an old config must still load after a field is renamed.
    """
    if not isinstance(data, dict):
        return cls()
    if not dataclasses.is_dataclass(cls):
        return data
    hints = _resolve_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        annotation = hints.get(field.name, field.type)
        optional, inner = _is_optional(annotation)
        origin = typing.get_origin(inner)
        if dataclasses.is_dataclass(inner) and isinstance(value, dict):
            kwargs[field.name] = build_dataclass(inner, value)
        elif origin in (list, tuple) and isinstance(value, (list, tuple)):
            args = typing.get_args(inner)
            if args and dataclasses.is_dataclass(args[0]):
                kwargs[field.name] = [build_dataclass(args[0], item) for item in value]
            elif origin is tuple:
                kwargs[field.name] = tuple(value)
            else:
                kwargs[field.name] = list(value)
        elif optional and value is None:
            kwargs[field.name] = None
        else:
            kwargs[field.name] = value
    return cls(**kwargs)


class ConfigFile:
    """Load/save a dataclass tree at ``path``, falling back to defaults."""

    def __init__(self, path, factory):
        self.path = Path(path)
        self.factory = factory

    def load(self):
        return build_dataclass(self.factory, load_json(self.path))

    def save(self, config) -> Path:
        return save_json(self.path, config)

    def exists(self) -> bool:
        return self.path.exists()

    def ensure(self):
        """Create the file with defaults when missing and return the config."""
        config = self.load()
        if not self.path.exists():
            self.save(config)
        return config


def env_path(variable: str, default) -> Path:
    return Path(os.environ.get(variable) or default)
