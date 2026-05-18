"""Canonical offline-RL package.

The legacy ``holosoma.agents.offline_cql`` package has been removed. New
runtime code lives under ``holosoma.agents.offline_rl``. Historical
checkpoint/config ``_target_`` metadata that names the removed package remains
supported through :mod:`holosoma.agents.offline_rl.common.target_compat` and
resolves directly to canonical agents without importing ``offline_cql``.
"""

__all__: list[str] = []
