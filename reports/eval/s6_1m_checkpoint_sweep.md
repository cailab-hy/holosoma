# S6 SMQR-SG 0–1M Checkpoint Sweep

Run: `exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096`  
Attempted steps: [50000, 100000, 200000, 300000, 500000, 750000, 1000000]  
Skipped: [{'step': 50000, 'reason': 'checkpoint missing'}, {'step': 100000, 'reason': 'checkpoint missing'}, {'step': 200000, 'reason': 'checkpoint missing'}]

## Metrics per checkpoint

| step | success_rate | mean_reward | mean_length | lift_rate_v1 | mean_min_hand_obj_dist | mean_goal_progress_frac |
|------|--------------|-------------|-------------|--------------|------------------------|-------------------------|
| 300000 | 0.0859 | 17.3380 | 177.14 | 0.3789 | 0.4041 | 0.2439 |
| 500000 | 0.0664 | 18.6854 | 190.64 | 0.4844 | 0.3757 | 0.2355 |
| 750000 | 0.1562 | 21.4134 | 215.43 | 0.4727 | 0.3793 | 0.2540 |
| 1000000 | 0.2500 | 26.6347 | 262.99 | 0.5703 | 0.3373 | 0.2554 |

## Best checkpoint

- step: **1000000**
- success_rate: 0.2500
- mean_reward:  26.6347
- lift_rate_v1: 0.5703
- mean_min_hand_obj_dist: 0.3373

Tie-break order: success_rate → mean_reward → lift_rate_v1 → -mean_min_hand_obj_dist → mean_goal_progress_frac
