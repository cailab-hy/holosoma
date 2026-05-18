"""SMQR-SG algorithm config alias (Step 4-C thin wrapper).

Per ``docs/offline_rl_refactor_plan.md`` §6 the dedicated config
dataclass migration is deferred to a later phase.  For now this module
simply re-exports the legacy :class:`OfflineCQLConfig` so downstream
code has a stable import point without touching ``config_types/algo.py``.
"""

from holosoma.config_types.algo import OfflineCQLConfig

SMQRSGAlgorithmConfig = OfflineCQLConfig

__all__ = ["SMQRSGAlgorithmConfig"]
