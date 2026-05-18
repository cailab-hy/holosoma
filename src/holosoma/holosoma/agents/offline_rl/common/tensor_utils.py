"""Tensor / batch utilities — Step 7-B (placeholder).

Batch sanity checks (dtype / finiteness / next_obs validation) and the
observation-key reconciliation block currently live inline inside
``OfflineCQLAgent.setup``.  Extraction into free helpers here is
scheduled for the Step 7-C / 7-D unified setup refactor.

This module exposes an empty public surface for now — importers can
still ``from offline_rl.common import tensor_utils`` without side
effects.  No behaviour change is intended.
"""

__all__: list[str] = []
