# Post-H predictive validity of A^H = G^H - b(kappa)

Dataset `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5`, sidecar `offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5.aw_weights.H50.vh.npz` (H=50, 20 progress bins, sigma(A)=0.3587, rhash verified). Primary window M=100; min rows per bin for the aggregate: 200; episode-cluster bootstrap with 50 resamples.

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
| 100 | 58.01 [56.87, 58.78] | 56.68 [55.48, 57.17] | 57.97 [56.84, 58.55] | 61.02 [60.14, 61.91] | no | 19 |

Count-weighted aggregate over bins with 95% episode-cluster bootstrap CI; unweighted mean over bins: M=100: 46.28 | 43.89 | 44.53 | 48.12.

## Logistic regression with progress-bin fixed effects: logit P(Y=1) = beta_A * A/sigma + gamma_bin

| M | beta_A (per 1 sigma of A) | 95% cluster CI | odds ratio per sigma | rows | episodes |
|---|---|---|---|---|---|
| 100 | 0.0619 | [0.0328, 0.0819] | 1.064 | 1,337,170 | 16,511 |

## Panel B: pooled quartiles, raw G^H (= any dataset-wide scalar baseline) vs A^H (primary M)

| signal | Q | P(bad tracking) % | mean phase | hazard wall bins [4, 5, 12, 13] share % | append share % |
|---|---|---|---|---|---|
| raw G^H | Q1 | 46.79 | 0.439 | 37.7 | 0.4 |
| raw G^H | Q2 | 68.77 | 0.360 | 1.5 | 4.1 |
| raw G^H | Q3 | 62.51 | 0.349 | 0.0 | 16.6 |
| raw G^H | Q4 | 55.61 | 0.267 | 0.0 | 28.2 |
| A^H | Q1 | 55.78 | 0.359 | 11.3 | 13.8 |
| A^H | Q2 | 58.04 | 0.332 | 6.1 | 14.9 |
| A^H | Q3 | 60.24 | 0.338 | 7.1 | 12.8 |
| A^H | Q4 | 59.61 | 0.385 | 14.8 | 7.9 |

## Caveats to state in the paper

- Rows whose episode ends before t+H (by bad tracking or by motion end) are excluded: a failure there is already inside the return that built A^H, and a motion end there leaves no post-H trajectory. This exclusion avoids using outcomes already contained in the return used to construct A^H.
- motion_ends inside the validation window is an observed negative (Y=0), not a censored row; only timeout / segment end / truncation / incomplete episodes are censored (excluded).
- Within a progress bin the quartiles of A^H and of raw G^H coincide, and beta_A is the same for both under bin fixed effects: Panel A and the regression test predictive information *after* conditioning on progress; they do not separate the progress-conditioned baseline from a global one. Panel B does that through quartile composition.
- Raw G^H of rows near the motion end is a truncated sum (fewer than H rewards), so part of raw-Q1's late-phase excess is mechanical truncation rather than difficulty; both effects are what the progress-conditioned baseline removes.
- The result is an association on the fixed dataset (transitions within an episode are correlated; CIs are episode-clustered). It shows A^H retains predictive information about downstream trajectory quality after progress normalisation; it does not show A^H is the true advantage.
