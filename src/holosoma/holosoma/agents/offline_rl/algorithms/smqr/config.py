"""SMQR algorithm config alias.

Step 4-B intentionally re-uses :class:`OfflineCQLConfig` verbatim.  A
dedicated dataclass split is deferred to a later phase (per
``docs/offline_rl_refactor_plan.md`` §6).  Centralising the alias here
gives downstream code a stable import point while preserving the
single source of truth for runtime defaults.
"""

from holosoma.config_types.algo import OfflineCQLConfig

SMQRAlgorithmConfig = OfflineCQLConfig

__all__ = ["SMQRAlgorithmConfig"]
