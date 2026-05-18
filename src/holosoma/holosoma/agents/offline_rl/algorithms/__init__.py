"""offline_rl.algorithms — Phase 1 placeholder package.

This sub-package is reserved for algorithm-specific implementations that will
be introduced in later phases of the offline-RL refactor:

    * Phase 2 — ``algorithms/cql/``      (CQL conservative penalty extraction)
    * Phase 3 — ``algorithms/smqr/``     (SMQR anchor / q_times_g logits)
    * Phase 4 — ``algorithms/smqr_sg/``  (SMQR-SG sg_weighted_lse / sg_blend / λ schedule)
    * Phase 5 — ``registry.py``          (algorithm registry)
    * Phase 6 — ``bc / iql / td3_bc / awac`` placeholders

In Phase 1 this package is intentionally empty. No registry, no agent
classes, no losses are defined here yet.
"""

__all__: list[str] = []
