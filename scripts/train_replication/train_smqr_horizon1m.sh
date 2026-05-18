#!/usr/bin/env bash
# SMQR (anchor-only, sc_tau_res_scale=0.0) — 1,000,000 step horizon validation.
#
# Goal
# ----
# Long-horizon validation of the hypothesis "Q ≈ tau_ref overlap occurs
# frequently throughout SMQR training" — extends the 100k pilot
# (logs/hv-g1-manager/exp_10_smqr_pilot100k_seed1) to 1M iterations.
#
# Strictly hypothesis-validation: no train-loss change, no scale sweep, no
# CQL counterpart in this script.
#
# Run name / log convention
# -------------------------
#   stable alias dir : logs/hv-g1-manager/exp_11_smqr_horizon1m_seed1/
#   ckpt cadence     : every 5,000 iters (covers ALL of
#                      {5k,10k,15k,20k,50k,100k,200k,500k,1000k})
#   total ckpt count : 200 files ≈ 200 × 5 MB ≈ 1.1 GB on disk
#   tee'd train log  : logs/train_replication/train_smqr_horizon1m.log
#
# Usage
# -----
#   mkdir -p logs/train_replication
#   bash scripts/train_replication/train_smqr_horizon1m.sh 2>&1 \
#     | tee logs/train_replication/train_smqr_horizon1m.log
#
#   DRY_RUN=1     bash scripts/train_replication/train_smqr_horizon1m.sh
#   LOGGER=disabled bash scripts/train_replication/train_smqr_horizon1m.sh
#   NUM_ITERS=500000 bash scripts/train_replication/train_smqr_horizon1m.sh   # 50% trial
#
# Resume option (only if you choose to bridge from the 100k pilot instead of
# starting fresh — NOT the default per the design doc):
#   RESUME_FROM=logs/hv-g1-manager/exp_10_smqr_pilot100k_seed1/model_0100000.pt \
#     bash scripts/train_replication/train_smqr_horizon1m.sh
#
# Guardrails
# ----------
# - No train loss code is modified.
# - No scale sweep: sc_tau_res_scale is hard-pinned to 0.0 (anchor-only).
# - No CQL run is launched here.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pin horizon defaults BEFORE sourcing _common.sh.
: "${NUM_ITERS:=1000000}"
: "${SAVE_INTERVAL:=5000}"   # 5k cadence covers every required milestone.
export NUM_ITERS SAVE_INTERVAL

# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SEED=1
ALGO_LABEL="smqr_cont_self_anchor_only"
RUN_TAG="exp_11_smqr_horizon1m_seed${SEED}"

extra_args=(
  --algo.config.critic-penalty-mode smqr_cont_self
  --algo.config.sc-tau-res-scale 0.0
)

# Optional resume bridge (off by default).
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "[horizon1m] ERROR: RESUME_FROM not found: ${RESUME_FROM}" >&2
    exit 4
  fi
  echo "[horizon1m] RESUME mode: bridging from ${RESUME_FROM}"
  echo "[horizon1m]   start_step will be auto-restored from ckpt's global_step."
  echo "[horizon1m]   total iters target = ${NUM_ITERS} (must exceed ckpt's step)."
  extra_args+=(--training.checkpoint "${RESUME_FROM}")
fi

run_training "${extra_args[@]}"
