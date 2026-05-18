"""Template scaffold for adding a new offline-RL baseline algorithm.

This module is a **placeholder** — calling any helper here raises
``NotImplementedError``.  Copy this package, rename, and follow
``README.md`` to plug in a real baseline.
"""

from holosoma.agents.offline_rl.algorithms.template.config import (
    TemplateAlgorithmConfig,
)
from holosoma.agents.offline_rl.algorithms.template.losses import (
    compute_template_loss,
)
from holosoma.agents.offline_rl.algorithms.template.agent import (
    TemplateOfflineAgent,
)

__all__ = [
    "TemplateAlgorithmConfig",
    "TemplateOfflineAgent",
    "compute_template_loss",
]
