"""Base config — Phase 1 compatibility wrapper.

This module is a compatibility wrapper for the Phase 1 offline-RL refactor.
It re-exports the legacy ``holosoma.config_types.algo.OfflineCQLConfig`` as
``OfflineRLBaseConfig`` so downstream code can begin migrating to the
algorithm-neutral name without any field or default change.

The original ``OfflineCQLConfig`` is **NOT** modified. The new name is bound
directly to the same class:

    >>> from holosoma.agents.offline_rl.common.config_base import OfflineRLBaseConfig
    >>> from holosoma.config_types.algo import OfflineCQLConfig
    >>> OfflineRLBaseConfig is OfflineCQLConfig
    True

No behaviour change is intended.
"""

from holosoma.config_types.algo import OfflineCQLConfig

# Algorithm-neutral alias. Bound directly to the legacy class so that
# isinstance / object identity / tyro round-tripping all behave identically.
OfflineRLBaseConfig = OfflineCQLConfig

__all__ = [
    "OfflineRLBaseConfig",
    "OfflineCQLConfig",
]
