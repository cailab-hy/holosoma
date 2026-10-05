# Post-H predictive validity of A^H = G^H - b(kappa)

Dataset `offline_data/g1_29dof_wbt_lafan_dance1_fastsac_1m_episode_env256_dataset.h5`, sidecar `offline_data/g1_29dof_wbt_lafan_dance1_fastsac_1m_episode_env256_dataset.h5.aw_weights.npz` (H=50, 20 progress bins, sigma(A)=0.9425, rhash verified). Primary window M=100; min rows per bin for the aggregate: 200; episode-cluster bootstrap with 200 resamples.

## Sample construction (primary M)

| item | rows |
|---|---|
| rows_total | 2,513,562 |
| episodes | 21,376 |
| excluded_episode_ends_before_t+H | 737,455 |
|   of which end reason bad_tracking | 509,941 |
|   of which end reason motion_ends | 227,214 |
|   of which end reason other | 300 |
| censored_in_window (timeout/segment/truncation/incomplete) | 600 |
| analysed | 1,775,507 |
|   y=1 (bad tracking in window) | 473,297 |
|   y=0 via motion_ends in window | 336,910 |
|   y=0 via window fully observed | 965,300 |

## Panel A: within-progress-bin quartiles of A^H -> P(bad tracking in post-H window) (%)

| M | Q1 (lowest A) | Q2 | Q3 | Q4 (highest A) | monotone | bins used |
|---|---|---|---|---|---|---|
| 100 | 51.37 [50.31, 52.44] | 30.53 [29.65, 31.46] | 16.06 [15.32, 16.84] | 8.67 [8.01, 9.17] | yes | 18 |
| 50 | 34.22 [33.51, 35.07] | 17.09 [16.46, 17.74] | 7.50 [7.06, 7.92] | 3.39 [3.09, 3.67] | yes | 18 |
| 150 | 60.41 [59.13, 61.69] | 39.63 [38.42, 40.79] | 23.17 [22.21, 24.09] | 13.46 [12.56, 14.28] | yes | 18 |

Count-weighted aggregate over bins with 95% episode-cluster bootstrap CI; unweighted mean over bins: M=100: 51.68 | 30.81 | 16.07 | 8.55; M=50: 34.38 | 17.31 | 7.57 | 3.40; M=150: 60.71 | 39.95 | 23.23 | 13.30.

## Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma + gamma_bin

| M | beta_A (per 1 sigma of A) | 95% cluster CI | odds ratio per sigma | rows | episodes |
|---|---|---|---|---|---|
| 100 | -2.4563 | [-2.5563, -2.3726] | 0.086 | 1,775,507 | 10,571 |
| 50 | -2.4332 | [-2.5131, -2.3584] | 0.088 | 1,775,807 | 10,571 |
| 150 | -2.4509 | [-2.5384, -2.3483] | 0.086 | 1,775,207 | 10,571 |

## Panel B: pooled quartiles, raw G^H (= any dataset-wide scalar baseline) vs A^H (primary M)

| signal | Q | P(bad tracking) % | mean phase | highest-hazard bins [6, 7] share % | append share % |
|---|---|---|---|---|---|
| raw G^H | Q1 | 46.06 | 0.546 | 12.4 | 0.0 |
| raw G^H | Q2 | 27.92 | 0.511 | 12.2 | 0.0 |
| raw G^H | Q3 | 18.09 | 0.473 | 10.5 | 0.0 |
| raw G^H | Q4 | 14.56 | 0.305 | 5.1 | 0.0 |
| A^H | Q1 | 47.01 | 0.542 | 7.7 | 0.0 |
| A^H | Q2 | 29.59 | 0.504 | 9.5 | 0.0 |
| A^H | Q3 | 18.15 | 0.463 | 10.6 | 0.0 |
| A^H | Q4 | 11.88 | 0.325 | 12.4 | 0.0 |

## Caveats to state in the paper

- Rows whose episode ends before t+H (by bad tracking or by motion end) are excluded: a failure there is already inside the return that built A^H, and a motion end there leaves no post-H trajectory. This exclusion avoids using outcomes already contained in the return used to construct A^H.
- motion_ends inside the validation window is an observed negative (Y=0), not a censored row; only timeout / segment end / truncation / incomplete episodes are censored (excluded).
- Within a progress bin the quartiles of A^H and of raw G^H coincide, and beta_A is the same for both under bin fixed effects: Panel A and the regression test predictive information *after* conditioning on progress; they do not separate the progress-conditioned baseline from a global one. Panel B does that through quartile composition.
- Raw G^H of rows near the motion end is a truncated sum (fewer than H rewards), so part of raw-Q1's late-phase excess is mechanical truncation rather than difficulty; both effects are what the progress-conditioned baseline removes.
- The result is an association on the fixed dataset (transitions within an episode are correlated; CIs are episode-clustered). It shows A^H retains predictive information about downstream trajectory quality after progress normalisation; it does not show A^H is the true advantage.
