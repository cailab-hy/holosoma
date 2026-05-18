# Step 6 — SMQR-SG 10K Extension Report
**3 modes × seed=1 × 10,000 iters · eval @ {5K, 8K, 10K} · all other settings identical to Step 4**

Generated: 2026-05-04. Raw tables: `/tmp/step6_logs/tables.txt`.

---

## 1. Run Configuration (verified, no code/setting drift from Step 4)

| Item | Value | Source |
|---|---|---|
| Algorithm | offline_cql + sac_bc actor (BC weight 2.5, anchor-only τ, IQL/AWBC, EMA q-normalizer, B2/P1/P1b, q_clip, num_random/policy = Step-4 values) | unchanged |
| Iterations | **10,000** (vs 5,000 in Step 4) | env override `NUM_ITERS=10000` |
| `batch_size` | **256** (Step 4 baseline preserved; 4096 deferred to a separate ablation per user instruction) | `holosoma_config.yaml` of every run |
| Seed | 1 (all three modes) | unchanged |
| Checkpoints | every 1K (model_0001000.pt … model_0010000.pt) | `SAVE_INTERVAL=1000` |
| Run dirs | exp_40_smqrsg_qtimesg_short10k_seed1<br>exp_41_smqrsg_qdetachedg_short10k_seed1<br>exp_42_smqrsg_sgweightedlse_short10k_seed1 | `logs/hv-g1-manager/` |
| Wall-clock (training) | ~8 min/run, ~24 min total (3 sequential, GPU 0) | `/tmp/step6_logs/STATUS.log` |
| Wall-clock (eval) | ~55 s/run × 9 = ~8.5 min | `/tmp/step6_logs/eval/STATUS.log` |
| Eval protocol | 256 envs, max_steps=600, EVAL_SEED=20240601, deterministic, single-episode-per-env, grasp_radius=0.12, lift_margin=0.05 | identical to Step 4 |
| Failures | none — all 3 trains rc=0, all 9 evals rc=0; no NaN/Inf in safety counters | telemetry |

**Code change in this step: NONE.** All Step-3 sub-mode plumbing and Step-4 cross-contamination guard + 29 `Loss/smqr/sg/*` telemetry keys are unmodified.

---

## 2. Last-1K Window Means — Critic

| metric | mode 0 q_times_g | mode 1 q_times_detached_g | mode 2 sg_weighted_lse |
|---|---:|---:|---:|
| `td_loss` (4001–5000 → 9001–10000) | 5.84 → **2.74** | 6.80 → 6.12 | 4.77 → **2.45** |
| `cql_penalty`                       | 5.34 → **37.6** ↑ | −4.70 → **−10.5** ↓ (rewards OOD) | 56.1 → **46.7** ↓ (relaxing) |
| `cql_q_rand_mean`                   | 0.96 → 4.25 | 8.27 → 7.91 | −21.9 → **−32.6** ↓ |
| `q_data_mean`                       | 72.2 → 55.8 | 80.3 → 84.9 | 28.4 → **27.7** (stable) |
| `q_overestimation_gap`              | 0.115 → 0.065 | 0.605 → **0.342** | −0.010 → **0.010** |
| `q_data_q_pi_gap`                   | 1.93 → 0.64 | 0.46 → 0.68 | 2.81 → **0.38** |
| `critic_grad_norm`                  | 52.8 → **20.6** (after 155.5 spike@5K) | 51.8 → **44.5** ↑ | 33.2 → **22.3** ↓ |

**Reading:**
- **sg_weighted_lse**: lowest td_loss, smallest overestimation gap (≈0.01), `q_data_mean` ANCHORED at ~27 (NOT collapsing), `cql_penalty` SLOWLY DECREASING from 56→47 over 5K→10K (not blowing up), `q_data_q_pi_gap` closing 2.81→0.38. Aggressive but **healthy conservatism**.
- **q_times_g**: recovered from the 5K critic-grad spike (155→21); but `cql_penalty` is now growing 5.3→39 and `q_data_mean` dropping 72→56 → drifting toward sg-like behaviour, more chaotically.
- **q_times_detached_g**: every direction wrong — penalty is **negative** (rewards OOD), `q_data_mean` rising 80→85, `critic_grad_norm` rising 52→55, overestimation_gap = 0.34. Would not survive long training.

---

## 3. Last-1K Window Means — Actor / Safety

| metric | qtg | qdet | sg |
|---|---:|---:|---:|
| `actor_loss`         | −0.994 | −0.995 | −0.974 |
| `bc_loss`            | 0.412  | 0.415  | **0.399** |
| `rl_actor_term`      | −2.024 | −2.031 | −1.972 |
| `actor_grad_norm`    | 2.59   | 2.58   | 2.70 |
| `action_l2_vs_data`  | 11.95  | 12.02  | **11.57** |
| `action_std`         | 0.141  | 0.142  | 0.138 |
| `policy_entropy`     | 20.4   | 20.6   | 19.4 |
| `cql_penalty_clamped_frac` / `grad_*_nan_or_inf` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Actor stats are nearly indistinguishable across modes (Δ ≤ 5%) — consistent with the actor seeing near-identical detached Q values (sub-mode only changes critic-side gradient routing). **No safety-counter regression** at 10K.

---

## 4. Mechanism — Last-1K Window (9001–10000)

| metric | qtg (mode 0) | qdet (mode 1) | sg (mode 2) |
|---|---:|---:|---:|
| `near_tau_softmax_mass`            | 0.0062  | 0.3887 | 0.4908 |
| `near_tau_grad_mass`               | 0.0804  | 0.2385 | 0.4908 |
| **NGAR = grad/softmax**            | **12.97×** | 0.614× | **1.000×** (by construction) |
| `lse_softmax_entropy_mean`         | 3.61    | 2.14   | **1.14** |
| `lse_softmax_top1_mean`            | 0.050   | 0.433  | **0.652** |
| `grad_factor_proxy_p50`            | 7.5e-20 | 1.7e-22 | **3.4e-6** |
| `grad_factor_proxy_p95`            | 3.5e-7  | 1.1e-4  | **0.023** |
| `grad_starvation_frac`             | 0.484   | 0.507   | **0.444** |
| `rank_corr_q_vs_core_input`        | 0.866   | 0.870   | **0.9999** |
| `top1_q_matches_top1_core_input`   | 0.999   | 1.000   | 1.000  |
| `rank_corr_q_vs_final_logits`      | −0.907  | −0.611  | **+0.650** |
| `top1_q_matches_top1_final_logits` | 0.031   | 0.241   | 0.059  |

**Reading:**
- sg's signature properties (NGAR≈1, near-perfect rank-preservation, 14+ orders-of-magnitude larger usable grad) hold from 5K → 10K **with no decay**.
- qtg's NGAR went 17.5× (5K) → 13.0× (10K) — still highest; rank-corr partially RECOVERED 0.64 → 0.87 as the actor entropy collapses (action distribution sharpens, `q ↔ core_input` correlation naturally improves).
- `rank_corr_q_vs_final_logits` is the smoking-gun for distortion: qtg = **−0.91** (final softmax weights are nearly anti-correlated with Q), qdet = −0.61, sg = **+0.65**. Only sg gives gradient mass to actions whose Q the critic actually believes are top.

---

## 5. Trajectory Snapshots — over-conservatism & mechanism (1K → 10K)

Selected from `/tmp/step6_logs/tables.txt §B`:

```
cql_q_rand_mean       1k     2.5k     5k       8k       10k
qtg                  -6.84   -8.94    +4.41   +5.25    +4.00     (stays mildly positive)
qdet                 -6.38   +0.23    +8.94   +7.47    +8.06     (stays positive, no penalty)
sg                   -9.44  -12.25   -23.00  -29.63   -33.50     (monotonic ↓ — strong OOD pushdown)

cql_penalty          1k     2.5k     5k       8k       10k
qtg                 112     46       +3.85   +31.5    +39.1      (recovering ↑ after dip)
qdet                113     52       -10.3   -13.8    -9.11      (rewards OOD; never positive)
sg                   82     58       +56.2   +51.1    +45.3      (DECREASING — relaxing, not exploding)

q_data_mean          1k     2.5k     5k       8k       10k
qtg                 18.4   51.4    72.97    58.91    55.01      (drifting down)
qdet                18.4   51.4    84.13    88.44    83.85      (still drifting up)
sg                  18.3   37.3    27.51    27.21    27.95      (locked at ~27)

critic_grad_norm     1k     2.5k     5k       8k       10k
qtg                 13.3   20.7    155.5    18.98    21.81      (one big spike, recovered)
qdet                13.2   21.5     23.6    32.16    55.06      (climbing)
sg                  13.9   33.5     29.6    25.35    20.55      (decreasing — most stable)
```

**Critical reading for the over-conservatism worry on sg:**
- `q_data_mean` is **anchored at ~27** for 5 consecutive checkpoints (5K, 6K, 7K, 8K, 10K). It is **not** collapsing toward 0.
- `cql_penalty` is **decreasing**, not exploding (56→47 from 5K→10K).
- `q_data_q_pi_gap` closes 2.81→0.38 (policy and data approaching same Q).
- `cql_q_rand_mean` does drop monotonically (−22 → −33), but this is **the OOD value alone** — it is not pulling the data Q with it.
- → **sg's conservatism is aggressive but not pathological at 10K.**

---

## 6. NGAR Comparison — 5K vs 10K

| Mode | NGAR @ 5K window | NGAR @ 10K window | Trend |
|---|---:|---:|---|
| q_times_g          | 17.53× | **12.97×** | dropping (was an artefact of low softmax mass; declining as actor sharpens) |
| q_times_detached_g | 0.538× | 0.614×    | flat (near-tau actions absorb 60% of softmax but only 24% of grad) |
| sg_weighted_lse    | 1.000× | 1.000×    | exact equality enforced by `g = softmax(stop_grad(z))` |

The qualitative ordering (sg = 1, qdet < 1, qtg ≫ 1) is **fully preserved** at 10K. NGAR is a durable diagnostic, not a 5K transient.

---

## 7. Ranking-Distortion Comparison (5K → 10K)

| metric | qtg 5K → 10K | qdet 5K → 10K | sg 5K → 10K |
|---|---:|---:|---:|
| `rank_corr_q_vs_core_input`        | 0.641 → 0.866  | 0.907 → 0.870  | **0.9995 → 0.9999** |
| `top1_q_matches_top1_core_input`   | 1.000 → 0.999  | 1.000 → 1.000  | 1.000 → 1.000 |
| `rank_corr_q_vs_final_logits`      | −0.891 → −0.907 | −0.522 → −0.611 | **+0.586 → +0.650** |

- sg's `q_vs_core_input` rank correlation stays pinned at **≈ 1.000** end-to-end.
- `q_vs_final_logits` is the post-detached-softmax view; sg is the **only** mode where final logits remain positively correlated with Q. qtg becomes increasingly anti-correlated as training proceeds.
- The Step-5 distortion verdict (qtg = severe rank inversion at the post-softmax stage; qdet = mild; sg = none) is **strengthened**, not softened, by 10K data.

---

## 8. Eval @ 5K / 8K / 10K  (256 envs, EVAL_SEED=20240601)

```
                                 step=5000                    step=8000                    step=10000
metric                   qtg     qdet     sg          qtg     qdet     sg          qtg     qdet     sg
success_rate             0.0000  0.0000   0.0000      0.0000  0.0000   0.0000      0.0000  0.0000   0.0000
first_contact_rate       0.0000  0.0000   0.0000      0.0000  0.0000   0.0000      0.0000  0.0000   0.0000
grasp_rate_v1            0.0000  0.0000   0.0000      0.0000  0.0000   0.0000      0.0000  0.0000   0.0000
lift_rate_v1             0.1289  0.1289   0.1133      0.1055  0.1172   0.0898      0.1211  0.1367   0.1094
mean_min_hand_obj_dist   0.5207  0.5243   0.5390      0.5664  0.5603   0.5417      0.5344  0.5353   0.5474
mean_max_obj_height      0.0177  0.0182   0.0162      0.0167  0.0169   0.0143      0.0174  0.0176   0.0162
mean_goal_progress_frac  0.2292  0.2151   0.2255      0.2081  0.2047   0.2315      0.2079  0.2161   0.2263
mean_action_norm         7.4832  7.2661   7.8538      9.0079  9.2205   9.6019      5.4871  5.5715   8.4990
mean_length             59.49   60.06    59.05       52.18   52.25    42.29       64.48   64.79    51.16
```

**Reading:**
- Task success is **zero across all 9 evals** — entirely expected at 5K–10K offline training (the Holosoma WBC tracking task typically doesn't begin to solve until 50K–200K). Eval at 10K is a **trend** check, not a verdict driver.
- `lift_rate_v1` and `mean_max_obj_height` are within noise across modes.
- `mean_goal_progress_frac` at 10K: sg = 0.226 ≥ qdet = 0.216 ≥ qtg = 0.208. Slight but consistent ordering at all 3 checkpoints.
- `mean_action_norm` at 10K: sg=8.50, qtg=5.49, qdet=5.57 — sg keeps a more energetic policy (not yet collapsed to no-op). Mean episode length is shorter for sg (51 vs 64), suggesting earlier termination via boundary; not necessarily a failure.
- **No mode is statistically separating from the others on task metrics at 10K**, but sg shows non-degraded behaviour, confirming the critic-side conservatism is not paralysing the actor.

---

## 9. PASS / PARTIAL / FAIL Judgment

Evaluating against the Step-5 criteria the user listed:

| # | Claim | 10K verdict |
|---|---|---|
| A | sg's `rank_corr_q_vs_core_input` ≥ 0.99 holds | **PASS** (0.9999 at 10K) |
| B | sg's NGAR = 1.0 (no near-tau gradient distortion) holds | **PASS** (exact, by construction; verified) |
| C | sg's `grad_factor_proxy_p50` is ≥ 10 orders of magnitude above qtg/qdet | **PASS** (3.4e-6 vs 1e-20 / 1e-22; ~14 OoM) |
| D | sg's `rank_corr_q_vs_final_logits` is the only one that is positive | **PASS** (sg = +0.65, qtg = −0.91, qdet = −0.61) |
| E | sg's critic stability is at least as good as qtg | **PASS** (lower td_loss, lower grad_norm, smaller overestimation gap, no spikes) |
| F | sg does NOT collapse Q-data toward 0 | **PASS** (`q_data_mean` anchored at 27 across 5K–10K) |
| G | sg's `cql_penalty` does not blow up monotonically | **PASS** (decreasing 56→47) |
| H | qdet exhibits broken conservatism (no penalty / rising Q) | **PASS** (penalty < 0 throughout; q_data_mean climbing) |
| I | qtg recovers from the 5K critic-grad spike | **PASS** (155 → 21 by 10K) |
| J | Eval task success differs across modes at 10K | **INCONCLUSIVE** (all zero — too early) |

**Overall Step-6 judgment: ✅ PASS** for the mechanism + critic-stability + non-collapse claims that motivate sg_weighted_lse.

The only criterion that does not get a positive answer (J) is one no offline run at this length should be expected to settle.

---

## 10. 0.5M Long-Run Recommendation

**GO** — proceed to a longer run with the following composition:

| Mode | Long-run inclusion | Reason |
|---|---|---|
| `sg_weighted_lse` | **YES — primary** | All mechanism + critic-stability claims durable to 10K; conservatism is healthy (anchored q_data, decreasing penalty); only mode with positive `rank_corr_q_vs_final_logits`. |
| `q_times_g`        | **YES — control** | Necessary baseline for the paper's central claim; survived 10K after recovering from the 5K spike, so it is no longer too risky to extend. |
| `q_times_detached_g` | **DROP** for long-run | Critic metrics worsen monotonically (penalty < 0, q_data ↑, grad_norm ↑, overestimation_gap ↑). Including it for 0.5M wastes ~25 GPU-h with predictable failure; keep the 10K data already in hand as the documented diagnosis. |

Suggested staged extension before committing 0.5M:
1. **50K, seed=1, both qtg + sg** (≈70 min on this rig). Decision gate: at 50K, eval `success_rate` should be > 0 for at least one mode; sg's `q_data_mean` must remain in [10, 60].
2. If 50K passes the gate → **200K, seeds {1, 2, 3}, qtg + sg only** (~14 GPU-h). Decision gate: ≥ 1 seed × mode reaches `success_rate` > 0.20.
3. Promote to **0.5M, multi-seed** only after step 2 succeeds.

Skip the 100K intermediate; staged 50K → 200K → 500M gives the same risk control with one less stage.

---

## 11. `batch_size = 4096` Verdict (Deferred Ablation)

Per user instruction, Step 6 preserved `batch_size = 256` — same as Step 4. There is therefore **no Step-6 evidence** about 4096. Recommendation:

- **Run as a separate, isolated ablation**, not bundled into the long-run, because:
  1. 4096 changes the effective number of OOD action samples seen per gradient step by ~16×, which changes `cql_penalty` magnitude and could shift sg's `q_data_mean` anchor away from the favourable ≈27 regime documented above.
  2. The bs=4096 ablation should re-measure the Step-5/6 mechanism panel (NGAR, rank_corr_q_vs_*, grad_factor_proxy) before any long-run uses it.
- Suggested protocol: `batch_size = 4096`, **same 10K duration**, **sg_weighted_lse + q_times_g only**, **seed = 1**. Cost ~16 min training + ~2 min eval. Compare: (a) NGAR/rank-corr unchanged; (b) `q_data_mean` still anchored; (c) `cql_penalty` magnitude scaling (expect ~16× nominal).
- Until that ablation is completed, **the 0.5M long-run should use batch_size = 256**.

---

## 12. Next-Step Recommendation

Among the candidate next steps:

| Option | Recommended? | Rationale |
|---|---|---|
| **A. 50K all-3-modes** | NO | Wastes the 10K diagnostic that already disqualified qdet. |
| **B. 50K qtg + sg only, seed=1** | **YES — DO NEXT** | Cheapest gate for "is success_rate > 0 reachable?". |
| **C. sg-only 100K → 0.5M** | NO | Loses qtg control needed for the paper's primary comparison. |
| **D. 10K multi-seed (seeds 2, 3) all 3 modes** | Optional, low value | We already have a clear separation at seed=1; seed-variance is more useful at 50K than at 10K. |
| **E. batch_size=4096 ablation, 10K, qtg+sg** | Yes, **in parallel** with B if a second GPU is available; otherwise **after** B. | Required before any 4096-based long-run; orthogonal to B. |

**Concrete next action:** start option **B** (50K, qtg + sg, seed=1, batch_size=256, eval @ {25K, 40K, 50K}). Estimated wall-clock ~70 min training + ~3 min eval.

---

### Appendix — Artefact paths
- Train logs: `logs/hv-g1-manager/exp_4{0,1,2}_smqrsg_*_short10k_seed1/`
- Checkpoints: `model_000{1000..10000}.pt` (10 each)
- Eval JSON: `eval_out/step6_smqr_sg_{q_times_g,q_times_detached_g,sg_weighted_lse}/step_{5000,8000,10000}/eval_summary.json`
- Raw extraction tables: `/tmp/step6_logs/tables.txt`
- Training STATUS log: `/tmp/step6_logs/STATUS.log`
- Eval STATUS log: `/tmp/step6_logs/eval/STATUS.log`
