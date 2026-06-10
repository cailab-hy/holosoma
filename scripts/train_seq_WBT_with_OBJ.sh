#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run_step() {
  local name="$1"
  shift
  echo "============================================================"
  echo "[SEQ-TRAIN] START: ${name}"
  echo "============================================================"
  "$@"
  echo "[SEQ-TRAIN] DONE : ${name}"
}

# 1) CQL
run_step "CQL" \
  python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-offline-rl \
    algo:offline-cql \
    --training.seed 1 \
    --training.name exp_cql_seed1 \
    --algo.config.dataset-path offline_data/fastsac_dataset.h5 \
    --algo.config.num-learning-iterations 1000000 \
    --algo.config.save-interval 50000 \
    --algo.config.batch-size 4096 \
    --algo.config.bc-weight 3.0 \
    --algo.config.cql-loss-scale 0.5 \
    --algo.config.cql-num-random-actions 10 \
    --algo.config.cql-num-policy-actions 10 \
    --algo.config.critic-penalty-mode vanilla_cql

# 2) SMQR
run_step "SMQR" \
  python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-offline-rl \
    algo:offline-smqr \
    --training.seed 1 \
    --training.name exp_smqr_seed1 \
    --algo.config.dataset-path offline_data/fastsac_dataset.h5 \
    --algo.config.num-learning-iterations 1000000 \
    --algo.config.save-interval 50000 \
    --algo.config.batch-size 4096 \
    --algo.config.bc-weight 3.0 \
    --algo.config.cql-loss-scale 0.5 \
    --algo.config.cql-num-random-actions 10 \
    --algo.config.cql-num-policy-actions 10 \
    --algo.config.critic-penalty-mode smqr_cont_self \
    --algo.config.algo-mode smqr_anchor \
    --algo.config.smqr-anchor-objective vanilla \
    --algo.config.sc-tau-res-scale 0.0 \
    --algo.config.smqr-lse-mode q_times_g

# 3) SMQR-SG
run_step "SMQR-SG" \
  python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-offline-rl \
    algo:offline-smqr-sg \
    --training.seed 1 \
    --training.name exp_smqr_sg_seed1 \
    --algo.config.dataset-path offline_data/fastsac_dataset.h5 \
    --algo.config.batch-size 4096 \
    --algo.config.num-learning-iterations 1000000 \
    --algo.config.save-interval 50000 \
    --algo.config.critic-penalty-mode smqr_cont_self \
    --algo.config.algo-mode smqr_anchor \
    --algo.config.smqr-anchor-objective vanilla \
    --algo.config.sc-tau-res-scale 0.0 \
    --algo.config.smqr-lse-mode sg_blend \
    --algo.config.smqr-blend-schedule fixed \
    --algo.config.smqr-blend-lambda-start 0.5 \
    --algo.config.smqr-blend-lambda-end 0.5 \
    --algo.config.cql-loss-scale 0.5 \
    --algo.config.bc-weight 3.0
    
echo "[SEQ-TRAIN] ALL DONE"
