"""Reproducibility plumbing: frozen fingerprints, run manifests, registry rows.

Two promises are kept here:

1. **A run is reproducible.**  Every output directory carries a manifest with
   the config fingerprint, the input fingerprints, the code commit and the
   environment, so the same numbers can be regenerated or the difference
   explained.
2. **A run is registered before it is believed.**  ``append_registry_row``
   writes into ``00_admin/experiment_registry.csv`` and is idempotent, so
   re-running an experiment updates its row instead of quietly adding a second
   one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REGISTRY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "created_at",
    "project",
    "research_question",
    "dataset_version",
    "train_end",
    "validation_period",
    "test_period",
    "holdout_used",
    "features",
    "model",
    "hyperparameters",
    "seed",
    "cost_config",
    "portfolio_config",
    "code_commit",
    "status",
    "result_path",
    "decision",
    "notes",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__} for a fingerprint")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def fingerprint(payload: Any) -> str:
    """Short stable hash of any JSON-serialisable object."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def fingerprint_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"cannot fingerprint a missing file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def git_commit(root: str | Path, *, fallback: str = "uncommitted") -> str:
    """Current commit, or 'uncommitted' when the tree has no commit yet.

    A run made on a dirty tree is recorded as ``<sha>+dirty``: the number can
    still be explained later.
    """

    directory = Path(root)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            return fallback
        sha = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            return f"{sha}+dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return fallback


def environment_snapshot() -> dict[str, str]:
    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module_name in ("numpy", "pandas"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - numpy/pandas are hard deps here
            snapshot[module_name] = "absent"
    return snapshot


def build_manifest(
    *,
    experiment_id: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, str | Path] | None = None,
    panel_fingerprint: str | None = None,
    root: str | Path = ".",
    extra: Mapping[str, Any] | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Assemble the record written next to every result."""

    config_fingerprint = fingerprint(config)
    input_records: dict[str, Any] = {}
    for name, path in (inputs or {}).items():
        item_path = Path(path)
        input_records[name] = {
            "path": str(item_path),
            "fingerprint": fingerprint_file(item_path) if item_path.exists() else None,
            "exists": item_path.exists(),
        }
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": dict(config),
        "config_fingerprint": config_fingerprint,
        "panel_fingerprint": panel_fingerprint,
        "code_commit": git_commit(root),
        "environment": environment_snapshot(),
        "inputs": input_records,
    }
    if extra:
        manifest["extra"] = dict(extra)
    manifest["manifest_fingerprint"] = fingerprint(
        {key: value for key, value in manifest.items() if key != "created_at_utc"}
    )
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return target


def read_registry(path: str | Path) -> list[dict[str, str]]:
    registry = Path(path)
    if not registry.exists():
        return []
    with registry.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_registry_row(
    path: str | Path,
    row: Mapping[str, Any],
    *,
    columns: Sequence[str] = REGISTRY_COLUMNS,
) -> str:
    """Insert or replace one experiment row.  Returns 'appended' or 'updated'.

    Replacement, not duplication: an experiment that is re-run with a fixed
    configuration must not appear twice in the registry, because a duplicated
    row is how a failed run quietly becomes a successful one.
    """

    registry = Path(path)
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise KeyError(f"registry row has unknown columns: {unknown}")
    missing = sorted(set(columns) - set(row))
    if missing:
        raise KeyError(f"registry row is missing required columns: {missing}")

    existing = read_registry(registry)
    experiment_id = str(row["experiment_id"])
    ordered_row = {column: row[column] for column in columns}
    replaced = False
    output: list[dict[str, Any]] = []
    for item in existing:
        if item.get("experiment_id") == experiment_id:
            output.append(ordered_row)
            replaced = True
        else:
            output.append(item)
    if not replaced:
        output.append(ordered_row)

    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for item in output:
            writer.writerow({column: item.get(column, "") for column in columns})
    return "updated" if replaced else "appended"
