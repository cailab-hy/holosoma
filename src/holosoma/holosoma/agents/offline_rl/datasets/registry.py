"""Offline-RL dataset registry.

Metadata-only lookup table. Dataset I/O is owned by the canonical
``holosoma.agents.offline_rl.common.datasets.OfflineDataset`` loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetEntry:
    """Metadata describing a registered offline-RL dataset."""

    name: str
    status: str   # 'legacy' | 'placeholder'
    description: str
    legacy_path: str | None = None
    observation_keys: tuple[str, ...] = field(default_factory=tuple)
    action_dim: int | None = None
    notes: str = ""


_ALLOWED_STATUS = {"legacy", "placeholder"}
_REGISTRY: dict[str, DatasetEntry] = {}


def _register(entry: DatasetEntry) -> None:
    if entry.status not in _ALLOWED_STATUS:
        raise ValueError(
            f"DatasetEntry.status must be one of {_ALLOWED_STATUS}, "
            f"got {entry.status!r}"
        )
    if entry.name in _REGISTRY:
        raise ValueError(f"dataset {entry.name!r} already registered")
    _REGISTRY[entry.name] = entry


# ── Legacy entry ────────────────────────────────────────────────────
_register(DatasetEntry(
    name="wbt_object",
    status="legacy",
    description=(
        "G1 29-DOF whole-body tracking with object manipulation "
        "(default offline dataset for CQL/SMQR/SMQR-SG)."
    ),
    legacy_path="offline_data/fastsac_dataset.h5",
    notes=(
        "Loaded via ``OfflineDataset`` (re-exported under "
        "``holosoma.agents.offline_rl.common.datasets.OfflineDataset``). "
        "Activated through the legacy preset "
        "``exp:g1-29dof-offline-rl``."
    ),
))

# ── Placeholder entry ──────────────────────────────────────────────
_register(DatasetEntry(
    name="locomotion_template",
    status="placeholder",
    description=(
        "Template for a future locomotion-only dataset (e.g. LAFAN, "
        "OMOMO walk subsets).  Not implemented in this phase."
    ),
    notes=(
        "See locomotion_template.py for the scaffolding entrypoint; "
        "schema and loader to be designed in a dedicated dataset "
        "integration phase."
    ),
))


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def get_dataset_entry(name: str) -> DatasetEntry:
    """Look up dataset metadata.  Raises ``KeyError`` if unknown.

    Placeholder entries return metadata; calling
    :func:`load_dataset` on them raises ``NotImplementedError``.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


def load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    """Construct a dataset instance (placeholder).

    Not wired in Step 5.  ``legacy`` entries raise
    ``NotImplementedError`` because the production loader lives in
    ``offline_cql.offline_cql`` and is invoked directly by
    ``train_agent.py``; ``placeholder`` entries raise
    ``NotImplementedError`` because the dataset has no loader yet.
    """
    entry = get_dataset_entry(name)
    if entry.status == "legacy":
        raise NotImplementedError(
            f"dataset {name!r} is loaded through the legacy "
            "OfflineDataset path "
            "(holosoma.agents.offline_rl.common.datasets.OfflineDataset); "
            "registry-based loading is not wired in this scaffold."
        )
    raise NotImplementedError(
        f"dataset {name!r} has status={entry.status!r} — "
        f"no loader available."
    )


__all__ = [
    "DatasetEntry",
    "get_dataset_entry",
    "list_datasets",
    "load_dataset",
]
