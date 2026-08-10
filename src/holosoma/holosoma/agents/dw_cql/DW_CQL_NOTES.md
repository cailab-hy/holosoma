# DW-CQL Placement Ablation

DW-CQL uses the exact AW-CQL sidecar signal and global mean-one normalization.
It changes only where that fixed transition weight is multiplied.

The agent owns a standalone copy of the scalar training loop and inherits
directly from `BaseAlgo`; it does not inherit from or modify `CQLAgent` or
`AWCQLAgent`. Shared actor/critic module definitions remain reusable model
components, but all sampling, sidecar alignment, losses, updates, logging,
checkpointing, and evaluation flow are contained in `dw_cql_agent.py`.

## Loss placement

- Critic TD: `mean(w * ((Q1 - y)^2 + (Q2 - y)^2))`.
- Conservative term: AW-CQL's existing `mean(w * (LSE - Q_data))` for both twins.
- Actor: `mean(w * (alpha_ent * log_pi - min(Q1, Q2)))`.
- Entropy-alpha tuning, CQL Lagrange tuning, Bellman target construction, target
  networks, twin aggregation, samplers, optimizers, and all hyperparameters are
  unchanged from the paired AW-CQL experiment.

## Registered design decisions

1. The actor uses the same row-wise `w(s, a_D)` as the sampled transition. No
   state-bin aggregation or additional state-weight model is introduced.
2. The precomputed global normalization is preserved. There is no per-batch
   renormalization; `dw_cql/batch_w_mean` is logged as a hygiene check.
3. The Bellman target `y = r + gamma * Q_target` is built unweighted. Weighting
   applies only to the completed residual.
4. Alpha, entropy target, Lagrange, twin handling, learning rates, update counts,
   and every other setting are inherited unchanged from AW-CQL.

## Entropy autotune boundary

The registered WBT and fall-and-get-up DW-CQL configs currently use automatic
entropy tuning. The actor objective is row-weighted in full, including its
`alpha_ent * log_pi` component, but the alpha controller remains explicitly
unweighted:

`L_alpha = mean(-exp(log_alpha) * stop_grad(log_pi + H_target))`.

This asymmetry is intentional. The sidecar is a transition-placement weight,
not a replacement target distribution for the entropy dual update. Weighting
`L_alpha` would make alpha respond to sampled sidecar composition and would add
a fourth placement beyond the registered three-position ablation.

This arm maps a Hong'23-style weight-placement pattern onto this project's
precomputed sidecar. It does not reproduce or evaluate that work's learned
weight construction.
