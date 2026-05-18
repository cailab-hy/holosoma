"""CQL algorithm config — Step 4-A thin wrapper.

This module is a *placeholder* for the eventual CQL-specific config split.
In Step 4-A no config keys are migrated yet; downstream code keeps using
``holosoma.config_types.algo.OfflineCQLConfig`` as before.

The alias below lets future code reference an algorithm-neutral name
without committing to a real schema split, and the ``is``-identity is
preserved so that tyro round-tripping / isinstance checks behave
identically.
"""

from holosoma.agents.offline_rl.common.config_base import OfflineRLBaseConfig
from holosoma.config_types.algo import OfflineCQLConfig

# Alias only. Bound directly to the legacy class so identity holds:
#   CQLAlgorithmConfig is OfflineCQLConfig is OfflineRLBaseConfig
CQLAlgorithmConfig = OfflineCQLConfig

__all__ = [
    "CQLAlgorithmConfig",
    "OfflineRLBaseConfig",
    "OfflineCQLConfig",
]
