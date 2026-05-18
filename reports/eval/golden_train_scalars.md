# Golden Train Scalar Sidecar Summary

Window: ±5000 steps around each sampled step.  
Sampled steps: 100K, 200K, 300K, 500K, 750K, 1M.


## Critic / actor scalars at final step

| run | step | td_loss | critic_loss | cql_penalty | q_data_mean | actor_loss | bc_loss | action_std |
|---|---|---|---|---|---|---|---|---|
| exp_80_perf_cql_seed1_bs4096_300k | 300000 | 0.0535 | 0.2239 | 21.3043 | 4.0986 | -0.4032 | 0.1899 | 0.1001 |
| exp_81_perf_smqr_qtimesg_seed1_bs4096_300k | 300000 | 0.0708 | 1.0991 | 128.5375 | 10.1949 | -0.6432 | 0.2029 | 0.1014 |
| exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096 | 1000000 | 0.0481 | 0.3598 | 77.9176 | 6.4124 | -0.5119 | 0.1807 | 0.0949 |

## SMQR-SG-specific scalars (S6 only)

| metric | window_mean@1M |
|---|---|
| near_tau_gradient_amplification_ratio | 0.9794039691195768 |
| near_tau_grad_mass | 0.9794039691195768 |
| rank_corr_q_vs_core_input | 0.779454393714082 |
| rank_corr_q_vs_final_logits | 0.8441768534043256 |
| smqr_blend_lambda_active | 0.5 |

## Missing tags per run (expected for non-SG runs)

- **exp_80_perf_cql_seed1_bs4096_300k**: ['near_tau_gradient_amplification_ratio', 'near_tau_grad_mass', 'rank_corr_q_vs_core_input', 'rank_corr_q_vs_final_logits', 'smqr_blend_lambda_active']
- **exp_81_perf_smqr_qtimesg_seed1_bs4096_300k**: ['smqr_blend_lambda_active']

Note: CQL/SMQR runs do not log SMQR-SG tags — `missing` is expected, not a failure.


Full per-step data in `golden_train_scalars.csv`.
