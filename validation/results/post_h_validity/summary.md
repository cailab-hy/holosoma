# Post-H predictive validity of A^H = G^H - b(kappa)

Dataset `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5`, sidecar `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.H50.npz` (H=50, 20 progress bins, sigma(A)=0.8832, rhash verified). Primary window M=100; min rows per bin for the aggregate: 200; episode-cluster bootstrap with 200 resamples.

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
| 100 | 63.33 [62.15, 64.57] | 57.47 [56.32, 58.67] | 56.82 [55.69, 57.86] | 56.06 [54.93, 57.36] | yes | 19 |
| 50 | 47.00 [46.06, 47.84] | 37.67 [36.96, 38.42] | 34.86 [34.05, 35.60] | 32.99 [32.17, 33.85] | yes | 19 |
| 150 | 68.54 [67.20, 69.80] | 63.34 [62.26, 64.52] | 63.01 [61.83, 64.07] | 64.89 [63.80, 66.19] | no | 19 |

Count-weighted aggregate over bins with 95% episode-cluster bootstrap CI; unweighted mean over bins: M=100: 50.72 | 44.57 | 43.66 | 43.88; M=50: 36.69 | 29.69 | 28.24 | 27.87; M=150: 57.23 | 51.61 | 50.73 | 51.43.

## Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma + gamma_bin

| M | beta_A (per 1 sigma of A) | 95% cluster CI | odds ratio per sigma | rows | episodes |
|---|---|---|---|---|---|
| 100 | -0.5184 | [-0.5931, -0.4531] | 0.595 | 1,337,170 | 16,511 |
| 50 | -0.8925 | [-0.9590, -0.8375] | 0.410 | 1,337,220 | 16,511 |
| 150 | -0.3707 | [-0.4668, -0.2959] | 0.690 | 1,337,120 | 16,511 |

## Panel B: pooled quartiles, raw G^H (= any dataset-wide scalar baseline) vs A^H (primary M)

| signal | Q | P(bad tracking) % | mean phase | wall-bin share % | append share % |
|---|---|---|---|---|---|
| raw G^H | Q1 | 46.79 | 0.439 | 37.7 | 0.4 |
| raw G^H | Q2 | 68.77 | 0.360 | 1.5 | 4.1 |
| raw G^H | Q3 | 62.51 | 0.349 | 0.0 | 16.6 |
| raw G^H | Q4 | 55.61 | 0.267 | 0.0 | 28.2 |
| A^H | Q1 | 59.26 | 0.388 | 10.4 | 16.4 |
| A^H | Q2 | 58.58 | 0.354 | 7.2 | 14.4 |
| A^H | Q3 | 59.16 | 0.340 | 6.8 | 13.8 |
| A^H | Q4 | 56.68 | 0.333 | 14.8 | 4.8 |

## Caveats to state in the paper

- Rows whose episode ends before t+H (by bad tracking or by motion end) are excluded: a failure there is already inside the return that built A^H, and a motion end there leaves no post-H trajectory. This exclusion avoids using outcomes already contained in the return used to construct A^H.
- motion_ends inside the validation window is an observed negative (Y=0), not a censored row; only timeout / segment end / truncation / incomplete episodes are censored (excluded).
- Within a progress bin the quartiles of A^H and of raw G^H coincide, and beta_A is the same for both under bin fixed effects: Panel A and the regression test predictive information *after* conditioning on progress; they do not separate the progress-conditioned baseline from a global one. Panel B does that through quartile composition.
- Raw G^H of rows near the motion end is a truncated sum (fewer than H rewards), so part of raw-Q1's late-phase excess is mechanical truncation rather than difficulty; both effects are what the progress-conditioned baseline removes.
- The result is an association on the fixed dataset (transitions within an episode are correlated; CIs are episode-clustered). It shows A^H retains predictive information about downstream trajectory quality after progress normalisation; it does not show A^H is the true advantage.
