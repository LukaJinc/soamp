"""YAML + pydantic config loading, with `_base_` composition.

A config file may declare `_base_: base.yaml` (path relative to its own
directory) and only specify the deltas; those deltas are deep-merged onto
the base file's dict before schema validation.
"""
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    base_dict = _load_yaml_with_base(path.parent / base_name)
    return _deep_merge(base_dict, raw)


def load_config(path: str | Path, schema_cls: Type[ConfigT]) -> ConfigT:
    """Load a YAML config (resolving `_base_` composition) and validate it
    against `schema_cls`. A relative `path` is resolved against the repo
    root, so callers don't need to re-derive `__file__`-relative paths."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    raw = _load_yaml_with_base(config_path)
    return schema_cls.model_validate(raw)
