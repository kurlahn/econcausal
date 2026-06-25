import pathlib
from typing import Any, Dict, List, Optional, cast

import dacite
import yaml

from econcausal.blueprint.schema import ExperimentCfg


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_mapping(path: str) -> Dict[str, Any]:
    here = pathlib.Path(path).resolve()
    with here.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    accumulated: Dict[str, Any] = {}
    for parent in parents:
        parent_path = (here.parent / parent).resolve()
        accumulated = _deep_merge(accumulated, load_mapping(str(parent_path)))
    return _deep_merge(accumulated, raw)


def _coerce(token: str) -> Any:
    lowered = token.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for converter in (int, float):
        try:
            return converter(token)
        except ValueError:
            continue
    if "," in token:
        return [_coerce(part) for part in token.split(",")]
    return token


def apply_overrides(mapping: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    result = dict(mapping)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.split(".")
        cursor: Dict[str, Any] = result
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _coerce(raw_value)
    return result


def build_experiment(path: str, overrides: Optional[List[str]] = None) -> ExperimentCfg:
    mapping = load_mapping(path)
    mapping = apply_overrides(mapping, list(overrides or []))
    structured = dacite.from_dict(
        data_class=ExperimentCfg,
        data=mapping,
        config=dacite.Config(cast=[tuple], strict=True),
    )
    return cast(ExperimentCfg, structured)
