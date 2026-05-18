"""Offline-RL dataset registry scaffold (Step 5).

This package is metadata-only.  The production dataset loader is the
canonical :class:`holosoma.agents.offline_rl.common.datasets.OfflineDataset`.
The removed ``offline_cql`` package no longer provides a dataset import path.

Future locomotion / multi-task datasets will be plugged in via
:func:`registry.get_dataset_entry`; this scaffold provides the
import surface.
"""

from holosoma.agents.offline_rl.datasets.registry import (
    DatasetEntry,
    get_dataset_entry,
    list_datasets,
)

__all__ = [
    "DatasetEntry",
    "get_dataset_entry",
    "list_datasets",
]
