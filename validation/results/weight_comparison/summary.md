# Weight signal comparison: PRe, V_H, ODPR-A

dataset `g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5`, N=2,525,605, episodes=34,914 (bad tracking 81.9%, motion completed 18.2%)

## 1. Distribution (all weights have mean 1)

| weights | ESS/N | std | min | p1 | p10 | p50 | p90 | p99 | max | share w<0.5 | share w>2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PRe | 0.716 | 0.629 | 0.000 | 0.02 | 0.19 | 0.97 | 1.81 | 2.63 | 6.1 | 25.9% | 6.1% |
| V_H | 0.495 | 1.010 | 0.000 | 0.02 | 0.27 | 0.77 | 1.73 | 7.06 | 7.1 | 23.5% | 7.6% |
| ODPR-A | 0.429 | 1.154 | 0.079 | 0.08 | 0.08 | 0.74 | 2.13 | 5.48 | 46.1 | 37.0% | 11.5% |

## 2. Agreement between weightings

| pair | Pearson | Spearman | within-bin Spearman | top-10% Jaccard | bottom-10% Jaccard |
|---|---|---|---|---|---|
| PRe vs V_H | 0.213 | 0.406 | 0.393 | 0.116 | 0.162 |
| PRe vs ODPR-A | 0.048 | 0.073 | 0.054 | 0.064 | 0.088 |
| V_H vs ODPR-A | 0.311 | 0.192 | 0.190 | 0.153 | 0.132 |

(top/bottom-10% Jaccard of two independent weightings would be 0.053.)

## 3. Per progress bin (K=20)

| bin | data % | bad-tracking hazard %/step | PRe mass % | V_H mass % | ODPR-A mass % | PRe mean w | V_H mean w | ODPR-A mean w | PRe std w | V_H std w | ODPR-A std w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12.10 | 0.43 | 14.52 | 12.70 | 16.94 | 1.199 | 1.049 | 1.399 | 0.676 | 1.159 | 1.663 |
| 1 | 7.46 | 0.46 | 7.23 | 7.02 | 7.41 | 0.968 | 0.941 | 0.993 | 0.469 | 0.826 | 1.072 |
| 2 | 8.02 | 0.42 | 7.44 | 7.47 | 6.77 | 0.928 | 0.931 | 0.844 | 0.470 | 0.754 | 0.911 |
| 3 | 8.34 | 0.60 | 7.77 | 7.95 | 6.45 | 0.931 | 0.952 | 0.773 | 0.603 | 0.815 | 0.775 |
| 4 | 6.70 | 3.30 | 5.87 | 6.08 | 5.27 | 0.876 | 0.907 | 0.787 | 0.585 | 0.843 | 0.772 |
| 5 | 2.82 | 4.55 | 3.28 | 2.99 | 2.67 | 1.164 | 1.061 | 0.945 | 1.079 | 1.102 | 1.128 |
| 6 | 2.53 | 1.10 | 2.60 | 2.75 | 2.80 | 1.029 | 1.085 | 1.107 | 0.605 | 1.219 | 1.451 |
| 7 | 3.34 | 0.82 | 3.40 | 3.70 | 3.77 | 1.018 | 1.107 | 1.129 | 0.588 | 1.220 | 1.190 |
| 8 | 3.99 | 0.99 | 4.19 | 4.71 | 4.44 | 1.049 | 1.180 | 1.111 | 0.664 | 1.289 | 1.142 |
| 9 | 4.31 | 1.07 | 4.67 | 5.41 | 4.76 | 1.084 | 1.254 | 1.105 | 0.720 | 1.425 | 1.142 |
| 10 | 4.59 | 0.97 | 4.68 | 5.05 | 4.69 | 1.019 | 1.100 | 1.022 | 0.650 | 1.210 | 1.003 |
| 11 | 4.59 | 1.13 | 4.39 | 4.73 | 4.20 | 0.957 | 1.029 | 0.913 | 0.603 | 1.024 | 0.906 |
| 12 | 4.61 | 1.47 | 4.02 | 4.34 | 3.89 | 0.872 | 0.941 | 0.843 | 0.531 | 0.843 | 0.768 |
| 13 | 3.26 | 4.52 | 2.93 | 3.19 | 2.91 | 0.898 | 0.979 | 0.893 | 0.679 | 0.972 | 1.019 |
| 14 | 2.00 | 2.54 | 2.13 | 2.19 | 2.40 | 1.065 | 1.095 | 1.200 | 0.799 | 1.190 | 1.662 |
| 15 | 2.63 | 1.09 | 2.84 | 2.91 | 3.41 | 1.079 | 1.109 | 1.299 | 0.685 | 1.292 | 1.515 |
| 16 | 3.35 | 0.55 | 3.37 | 3.34 | 3.87 | 1.006 | 0.996 | 1.155 | 0.514 | 1.079 | 1.377 |
| 17 | 4.28 | 0.47 | 4.24 | 4.22 | 4.27 | 0.990 | 0.984 | 0.996 | 0.472 | 1.000 | 1.161 |
| 18 | 5.22 | 0.38 | 5.19 | 4.76 | 4.41 | 0.994 | 0.913 | 0.845 | 0.650 | 0.770 | 0.906 |
| 19 | 5.83 | 0.37 | 5.25 | 4.50 | 4.68 | 0.900 | 0.771 | 0.802 | 0.583 | 0.341 | 0.882 |

| weights | Var(w) explained by bin (R2) | corr(bin mean w, bin hazard) | weight mass on the 4 highest-hazard bins (data share 14.8%) |
|---|---|---|---|
| PRe | 0.0251 | +0.007 | 14.21% |
| V_H | 0.0109 | +0.072 | 14.45% |
| ODPR-A | 0.0300 | -0.174 | 13.25% |

Highest-hazard bins: [5, 13, 4, 14].

## 4. What each weight correlates with (Spearman)

| weights | reward r_t | G^H (H-step return) | steps until episode end | episode completes motion (AUC) | within-bin: G^H | within-bin: episode completes (AUC) |
|---|---|---|---|---|---|---|
| PRe | +0.420 | +0.754 | +0.695 | 0.626 | +1.000 | 0.741 |
| V_H | +0.058 | +0.276 | +0.276 | 0.552 | +0.393 | 0.588 |
| ODPR-A | +0.015 | +0.072 | +0.098 | 0.507 | +0.054 | 0.518 |

AUC = probability that a transition from an episode that completes the motion gets a higher weight than one from an episode that ends by bad tracking (0.5 = no information).

## 5. Transitions that are about to terminate by bad tracking

`fail<=K`: the episode ends by bad tracking within the next K steps (K counted from the transition). Mean weight of those rows (1.0 = dataset mean), and AUC of the weight for separating them from all other rows (AUC < 0.5 means the weighting DOWN-weights soon-to-fail transitions; within-bin AUC removes the effect of where in the motion they are).

| K | share of rows | PRe mean w | V_H mean w | ODPR-A mean w | PRe AUC | V_H AUC | ODPR-A AUC | PRe within-bin AUC | V_H within-bin AUC | ODPR-A within-bin AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1% | 0.204 | 0.757 | 1.616 | 0.098 | 0.408 | 0.569 | 0.035 | 0.429 | 0.613 |
| 5 | 5.7% | 0.212 | 0.708 | 0.959 | 0.084 | 0.376 | 0.471 | 0.028 | 0.395 | 0.510 |
| 10 | 10.5% | 0.229 | 0.685 | 0.901 | 0.072 | 0.353 | 0.464 | 0.024 | 0.370 | 0.499 |
| 25 | 22.5% | 0.313 | 0.689 | 0.882 | 0.062 | 0.327 | 0.460 | 0.029 | 0.324 | 0.483 |
| 50 | 37.1% | 0.500 | 0.782 | 0.897 | 0.112 | 0.324 | 0.456 | 0.066 | 0.301 | 0.473 |
| 100 | 57.3% | 0.765 | 0.939 | 0.944 | 0.238 | 0.408 | 0.466 | 0.187 | 0.382 | 0.477 |
| 200 | 71.5% | 0.910 | 0.985 | 1.001 | 0.350 | 0.445 | 0.489 | 0.249 | 0.411 | 0.482 |

## 6. Failure AFTER the H=50 window (rows whose episode is still alive at t+H; 52.9% of rows)

Outcome: bad tracking within [t+H, t+H+M). This cannot be read off the H-step return itself.

| M | failure rate | PRe mean w (fail / survive) | V_H mean w (fail / survive) | ODPR-A mean w (fail / survive) | PRe within-bin AUC | V_H within-bin AUC | ODPR-A within-bin AUC |
|---|---|---|---|---|---|---|---|
| 50 | 38.1% | 1.251 / 1.414 | 1.227 / 1.156 | 1.031 / 1.159 | 0.381 | 0.507 | 0.492 |
| 100 | 58.4% | 1.333 / 1.379 | 1.202 / 1.156 | 1.101 / 1.124 | 0.427 | 0.515 | 0.494 |
| 150 | 64.9% | 1.353 / 1.350 | 1.205 / 1.142 | 1.113 / 1.105 | 0.449 | 0.527 | 0.499 |

## 7. Where the weight mass goes

| weights | mass on rows of completing episodes (data share 24.7%) | mass on the last 25 steps before a bad-tracking end (data share 22.5%) |
|---|---|---|
| PRe | 29.6% | 7.0% |
| V_H | 25.4% | 15.5% |
| ODPR-A | 24.3% | 19.8% |
