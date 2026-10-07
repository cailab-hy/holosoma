#!/usr/bin/env bash
# Step 0 for the double-kick motion: FastSAC episode-data collection (10k iterations, 256 recording envs)
# that writes the offline dataset every kick2 offline arm trains on, then computes the AW sidecar and the
# per-progress-bin hazard profile used to judge motion difficulty. Single seed.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1}"
FINAL_STEP="${FINAL_STEP:-10000}"
activate_env
ENTRY=src/holosoma/holosoma/train_agent.py run_seeds g1_29dof_wbt_lafan_kick2_fast_sac_episode_data g1-29dof-wbt-lafan-kick2-fast-sac-episode-data
if [[ -f "${KICK2_H5}" ]]; then
  ensure_aw_sidecar "${KICK2_H5}" 50 "${KICK2_H5}.aw_weights.npz"
  python scripts/analyze_h5_phase_failure_hazard.py "${KICK2_H5}" 2>/dev/null || true
fi
finish
