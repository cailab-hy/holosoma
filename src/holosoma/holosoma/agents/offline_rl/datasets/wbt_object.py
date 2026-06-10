"""WBT-object dataset metadata wrapper.

The dataset loader is canonically exposed as
``holosoma.agents.offline_rl.common.datasets.OfflineDataset``. The removed
``offline_cql`` package no longer provides a dataset-loader import path.
"""

from __future__ import annotations

# Re-export the canonical loader for stable import paths.
from holosoma.agents.offline_rl.common.datasets import OfflineDataset


# ── Static metadata ───────────────────────────────────────────────
WBT_OBJECT_METADATA = {
    "name": "wbt_object",
    "legacy_path": "offline_data/fastsac_dataset.h5",
    "format": "HDF5",
    "preset": "exp:g1-29dof-offline-rl",
    "robot": "G1 29-DOF",
    "task": "Whole-body tracking with object manipulation",
    "loader_class": "OfflineDataset",
    "loader_module": "holosoma.agents.offline_rl.common.datasets",
}


def get_loader_class():
    """Return the canonical loader class (alias).

    Provided so future code can do ``Loader = get_loader_class()``
    without hard-coding the legacy import path.
    """
    return OfflineDataset


__all__ = ["OfflineDataset", "WBT_OBJECT_METADATA", "get_loader_class"]
