"""Locomotion dataset template (Step 5 placeholder).

This module is a scaffold for adding locomotion-only offline datasets
(e.g. LAFAN walk subsets, OMOMO subsets).  No loader is implemented
yet — all entrypoints raise ``NotImplementedError``.

To enable a real locomotion dataset:
  1. Copy this module to ``datasets/<your_dataset>.py``.
  2. Implement ``LocomotionDataset.__init__`` (HDF5/torch-tensor load).
  3. Wire the dataset registry entry status from ``placeholder`` to
     ``legacy`` (or ``available``).
  4. Add an eval manifest entry and a golden smoke checkpoint.
"""

from __future__ import annotations

from typing import Any


class LocomotionDataset:
    """Placeholder locomotion dataset.  Not implemented."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LocomotionDataset is a placeholder — no loader is "
            "implemented in Step 5.  Replace this class with a "
            "real implementation before training."
        )


def load_locomotion_dataset(*args: Any, **kwargs: Any) -> Any:
    """Placeholder loader entrypoint."""
    raise NotImplementedError(
        "load_locomotion_dataset is a placeholder; see "
        "datasets/README.md for the integration checklist."
    )


__all__ = ["LocomotionDataset", "load_locomotion_dataset"]
