"""Logging helpers — Phase 1 placeholder.

This module is a compatibility wrapper placeholder for the Phase 1
offline-RL refactor. It is intentionally empty in Phase 1.

TensorBoard scalar emission currently happens inline inside
``OfflineCQLAgent.learn`` and ``OfflineCQLAgent._update_critic`` (rich-metrics
block). All tag names (e.g. ``Loss/td_loss``, ``Loss/cql_penalty``,
``Loss/smqr/sg/near_tau_grad_mass``, ``Loss/smqr_blend_lambda_active``) are
**frozen** for Phase 1 and must remain byte-identical so that the Step-1
extractor (``scripts/eval/extract_train_scalars.py``) keeps working without
any alias-map change.

No behaviour change is intended.
"""

__all__: list[str] = []
