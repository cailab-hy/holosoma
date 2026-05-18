"""offline_rl.algorithms.cql — Step 4-A scaffold.

This sub-package extracts the CQL-specific pieces of the offline-RL
implementation. As of Step 4-A only the vanilla CQL importance-weighted
logsumexp penalty has been factored out (see :mod:`losses`).

The agent class itself still lives at
``holosoma.agents.offline_cql.offline_cql_agent.OfflineCQLAgent``; only the
inner-loop loss computation has been moved here. No behaviour change is
intended.
"""

from holosoma.agents.offline_rl.algorithms.cql.losses import (
    compute_cql_logsumexp_penalty,
)

__all__ = [
    "compute_cql_logsumexp_penalty",
]
