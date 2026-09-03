"""
YAML configuration loading and small path/format helpers.

Nothing in this module imports torch, pyserini or beir, so it stays cheap to
import from anywhere in the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

VALID_STAGE_MODES = {"auto", "force", "skip"}


class _SafeDict(dict):
    """format_map() helper: unknown placeholders are left untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return cfg


def deep_get(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def require(cfg: Mapping[str, Any], dotted_key: str) -> Any:
    value = deep_get(cfg, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required YAML key: {dotted_key}")
    return value


def stage_mode(cfg: Mapping[str, Any], key: str) -> str:
    value = str(require(cfg, key)).lower()
    if value not in VALID_STAGE_MODES:
        raise ValueError(
            f"{key} must be one of {sorted(VALID_STAGE_MODES)}, got {value!r}"
        )
    return value


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def expand(text: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(text)))


def format_template(template: Any, **values: Any) -> str:
    """
    Like str.format(), but unknown placeholders are preserved instead of
    raising KeyError. This lets templates mention {dataset}/{lang}/{model}
    without every caller having to supply all of them.
    """
    return str(template).format_map(_SafeDict({k: str(v) for k, v in values.items()}))


def format_path(template: Any, **values: Any) -> Path:
    return Path(expand(format_template(template, **values)))
