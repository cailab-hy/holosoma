# Step 6 — Golden Eval Reproduction (post Step 4-A/B/C + Step 5)

Eval re-run with `--force` under current code state.


## Comparison vs. expected (tolerance: success/lift ±0.02 abs, reward/dist ±10% rel)

| run | metric | actual | expected | err | OK |
|---|---|---|---|---|---|
| cql_golden_300k | success_rate | 0 | 0 | 0.000000 | ✅ |
| cql_golden_300k | mean_reward | 11.3094 | 11.31 | 0.000049 | ✅ |
| cql_golden_300k | mean_length | 119.309 | 119.31 | 0.000012 | ✅ |
| cql_golden_300k | lift_rate_v1 | 0.34375 | 0.3438 | 0.000050 | ✅ |
| cql_golden_300k | mean_min_hand_obj_dist | 0.442461 | 0.4425 | 0.000088 | ✅ |
| smqr_anchor_golden_300k | success_rate | 0.09375 | 0.0938 | 0.000050 | ✅ |
| smqr_anchor_golden_300k | mean_reward | 17.514 | 17.514 | 0.000001 | ✅ |
| smqr_anchor_golden_300k | mean_length | 180.438 | 180.44 | 0.000014 | ✅ |
| smqr_anchor_golden_300k | lift_rate_v1 | 0.429688 | 0.4297 | 0.000013 | ✅ |
| smqr_anchor_golden_300k | mean_min_hand_obj_dist | 0.411952 | 0.412 | 0.000116 | ✅ |
| smqr_sg_current_best_1m | success_rate | 0.25 | 0.25 | 0.000000 | ✅ |
| smqr_sg_current_best_1m | mean_reward | 26.6347 | 26.6347 | 0.000002 | ✅ |
| smqr_sg_current_best_1m | mean_length | 262.988 | 262.99 | 0.000007 | ✅ |
| smqr_sg_current_best_1m | lift_rate_v1 | 0.570312 | 0.5703 | 0.000012 | ✅ |
| smqr_sg_current_best_1m | mean_min_hand_obj_dist | 0.337283 | 0.3373 | 0.000049 | ✅ |

## Required metrics present (all runs)

- cql_golden_300k: missing=[]
- smqr_anchor_golden_300k: missing=[]
- smqr_sg_current_best_1m: missing=[]