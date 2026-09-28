# Pre-registered predictions: common-Q level residual (written 2026-09-19, before any delta run)

Objective: P_res = w (LSE - Q_D) + 2 delta w m(s), m(s) = mean of Q over the dataset action and all LSE
candidate actions (gradient kept). delta = 0 is AW-CQL exactly. Common shift Q -> Q + c(s) changes P_res by
2 delta w c(s); the common-direction gradient is 2 delta w, spread uniformly over the N+1 Q outputs.
The AW relative term w(LSE - Q_D) is untouched, so the transition preference is identical for every delta.

| delta | Residual/common_force_mean (= 2 delta) | predicted Q_level vs AW (paired seed) |
| --- | --- | --- |
| -0.25 | -0.5 | Q_level > AW (upward pressure) |
| -0.10 | -0.2 | Q_level > AW, smaller |
| 0 (AW-CQL H50) | 0 | reference |
| +0.10 | +0.2 | Q_level < AW, smaller |
| +0.25 | +0.5 | Q_level < AW (downward pressure) |

Key figure: DeltaQ(run, t) = Q_level(run, seed s, t) - Q_level(AW-CQL H50, seed s, t) on the fixed probe set
(4096 transitions, probe seed 12345; scripts/probe_q_level.py --paired-baseline g1_29dof_wbt_aw_cql_H50).
Expected: sign(DeltaQ) = -sign(delta), |DeltaQ| monotone in |delta| (dose-response).

Earlier evidence (2026-09-19): the previous family w[(1+delta)LSE - (1-delta)Q_D] at delta=+0.5 collapsed
(uniform sinking, TD targets to -213) and its candidate-mean "centered" control at +0.5 diverged upward
because the centering lifted random-action Qs. |delta| <= 0.25 was chosen so the residual stays trainable.

Secondary: does the sign-dependent displacement come with a change in final-10 / last-best / h_pi(5)?
If control performance is flat while Q_level moves, the shared bracket's performance is level-invariant;
if delta > 0 degrades like CQL, the downward level force itself is harmful.

Runs: 4 deltas x 3 seeds = 12 (delta=0 reuses AW-CQL H50 seeds 1-3), seed-major order.
