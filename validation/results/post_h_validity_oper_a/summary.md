# Post-H predictive validity of A^H = G^H - b(kappa)

Dataset `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5`, sidecar `validation/results/oper_vs_aw/oper_a_pseudo_sidecar.npz` (H=50, 20 progress bins, sigma(A)=0.1397, rhash verified). Primary window M=100; min rows per bin for the aggregate: 200; episode-cluster bootstrap with 200 resamples.

## Sample construction (primary M)

| item | rows |
|---|---|
| rows_total | 2,525,605 |
| episodes | 34,914 |
| excluded_episode_ends_before_t+H | 1,188,335 |
|   of which end reason bad_tracking | 937,562 |
|   of which end reason motion_ends | 250,723 |
|   of which end reason other | 50 |
| censored_in_window (timeout/segment/truncation/incomplete) | 100 |
| analysed | 1,337,170 |
|   y=1 (bad tracking in window) | 781,183 |
|   y=0 via motion_ends in window | 224,375 |
|   y=0 via window fully observed | 331,612 |

## Panel A: within-progress-bin quartiles of A^H -> P(bad tracking in post-H window) (%)

| M | Q1 (lowest A) | Q2 | Q3 | Q4 (highest A) | monotone | bins used |
|---|---|---|---|---|---|---|
| 100 | 59.93 [59.03, 60.80] | 57.61 [56.57, 58.58] | 57.44 [56.57, 58.35] | 58.70 [57.83, 59.60] | no | 19 |
| 50 | 40.20 [39.48, 40.85] | 36.80 [36.17, 37.47] | 36.65 [36.05, 37.33] | 38.87 [38.32, 39.46] | no | 19 |
| 150 | 66.08 [65.13, 67.06] | 64.04 [62.92, 65.02] | 64.16 [63.19, 65.19] | 65.51 [64.63, 66.38] | no | 19 |

Count-weighted aggregate over bins with 95% episode-cluster bootstrap CI; unweighted mean over bins: M=100: 47.07 | 45.12 | 44.89 | 45.75; M=50: 32.13 | 30.05 | 29.72 | 30.60; M=150: 53.95 | 52.10 | 52.02 | 52.93.

## Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma + gamma_bin

| M | beta_A (per 1 sigma of A) | 95% cluster CI | odds ratio per sigma | rows | episodes |
|---|---|---|---|---|---|
| 100 | -0.0428 | [-0.0468, -0.0384] | 0.958 | 1,337,170 | 16,511 |
| 50 | -0.0460 | [-0.0494, -0.0426] | 0.955 | 1,337,220 | 16,511 |
| 150 | -0.0228 | [-0.0263, -0.0193] | 0.977 | 1,337,120 | 16,511 |

## Panel B: pooled quartiles, raw G^H (= any dataset-wide scalar baseline) vs A^H (primary M)

| signal | Q | P(bad tracking) % | mean phase | hazard wall bins [4, 5, 12, 13] share % | append share % |
|---|---|---|---|---|---|
| raw G^H | Q1 | 58.68 | 0.425 | 26.4 | 2.0 |
| raw G^H | Q2 | 64.65 | 0.362 | 9.8 | 5.7 |
| raw G^H | Q3 | 54.98 | 0.364 | 3.0 | 19.9 |
| raw G^H | Q4 | 55.38 | 0.263 | 0.0 | 21.7 |
| A^H | Q1 | 60.52 | 0.341 | 9.8 | 12.0 |
| A^H | Q2 | 57.50 | 0.358 | 8.5 | 14.5 |
| A^H | Q3 | 57.49 | 0.361 | 9.3 | 12.9 |
| A^H | Q4 | 58.17 | 0.355 | 11.6 | 9.9 |

## Caveats to state in the paper

- Rows whose episode ends before t+H (by bad tracking or by motion end) are excluded: a failure there is already inside the return that built A^H, and a motion end there leaves no post-H trajectory. This exclusion avoids using outcomes already contained in the return used to construct A^H.
- motion_ends inside the validation window is an observed negative (Y=0), not a censored row; only timeout / segment end / truncation / incomplete episodes are censored (excluded).
- Within a progress bin the quartiles of A^H and of raw G^H coincide, and beta_A is the same for both under bin fixed effects: Panel A and the regression test predictive information *after* conditioning on progress; they do not separate the progress-conditioned baseline from a global one. Panel B does that through quartile composition.
- Raw G^H of rows near the motion end is a truncated sum (fewer than H rewards), so part of raw-Q1's late-phase excess is mechanical truncation rather than difficulty; both effects are what the progress-conditioned baseline removes.
- The result is an association on the fixed dataset (transitions within an episode are correlated; CIs are episode-clustered). It shows A^H retains predictive information about downstream trajectory quality after progress normalisation; it does not show A^H is the true advantage.
