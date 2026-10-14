from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    return cfg


def apply_overrides(
    cfg: dict[str, Any],
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not overrides:
        return cfg

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key=value', got: {ov!r}")

        key_path, value_str = ov.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the parent dict
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        # Auto-cast the value
        d[keys[-1]] = _auto_cast(value_str)

    return cfg


def _auto_cast(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
