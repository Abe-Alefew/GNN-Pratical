"""
Model rollback utility (Assignment 4, Stage 4.9).

Restores the previous approved model version by moving the ``champion``
alias back to the prior version.  Logs the rollback reason.

For simplicity, this uses a JSON-based version registry (no MLflow
registry required, though MLflow is supported).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_FILE = "model_registry.json"


def _load_registry(registry_path: str) -> dict:
    """Load or initialise the version registry."""
    path = Path(registry_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"versions": [], "champion": None, "history": []}


def _save_registry(registry: dict, registry_path: str) -> None:
    """Persist the registry to disk."""
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_version(
    version_id: str,
    package_dir: str,
    metrics: dict,
    registry_path: str = REGISTRY_FILE,
) -> None:
    """Register a new model version.

    Args:
        version_id: Unique version identifier (e.g. MLflow run ID).
        package_dir: Path to the model package directory.
        metrics: Validation metrics dict.
        registry_path: Path to the JSON registry file.
    """
    registry = _load_registry(registry_path)
    registry["versions"].append({
        "version_id": version_id,
        "package_dir": package_dir,
        "metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_registry(registry, registry_path)


def promote_version(
    version_id: str,
    reason: str,
    registry_path: str = REGISTRY_FILE,
) -> None:
    """Set a version as the champion (approved model).

    Args:
        version_id: Version to promote.
        reason: Why this version was promoted.
        registry_path: Path to the JSON registry.
    """
    registry = _load_registry(registry_path)
    old_champion = registry.get("champion")

    registry["champion"] = version_id
    registry["history"].append({
        "action": "promote",
        "version_id": version_id,
        "previous_champion": old_champion,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_registry(registry, registry_path)


def rollback(
    reason: str = "Manual rollback",
    registry_path: str = REGISTRY_FILE,
) -> str | None:
    """Roll back to the previous champion version.

    Args:
        reason: Reason for the rollback.
        registry_path: Path to the JSON registry.

    Returns:
        The restored version ID, or None if no previous version exists.
    """
    registry = _load_registry(registry_path)

    # Find the most recent promotion before the current one
    promote_events = [
        h for h in registry["history"]
        if h["action"] == "promote"
    ]

    if len(promote_events) < 2:
        print("No previous version to roll back to.")
        return None

    # The second-to-last promotion is the previous champion
    previous = promote_events[-2]["version_id"]
    current = registry.get("champion")

    registry["champion"] = previous
    registry["history"].append({
        "action": "rollback",
        "from_version": current,
        "to_version": previous,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_registry(registry, registry_path)

    print(f"Rolled back: {current} -> {previous} (reason: {reason})")
    return previous


def get_champion(registry_path: str = REGISTRY_FILE) -> dict | None:
    """Get the current champion version info.

    Returns:
        Version dict or None if no champion is set.
    """
    registry = _load_registry(registry_path)
    champion_id = registry.get("champion")
    if champion_id is None:
        return None

    for v in registry["versions"]:
        if v["version_id"] == champion_id:
            return v
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.a4_mlops.rollback <reason>")
        sys.exit(1)

    reason = " ".join(sys.argv[1:])
    result = rollback(reason=reason)
    sys.exit(0 if result else 1)
