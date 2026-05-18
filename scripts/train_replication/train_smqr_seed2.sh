#!/usr/bin/env bash
# Replication run 3-C: SMQR (anchor-only, sc_tau_res_scale=0.0), seed=2, 20k iters.
#
# Same offline dataset & preset as the seed=1 baseline (exp_08_smqr).
# Output directory will be renamed to  logs/hv-g1-manager/exp_09_smqr_seed2/
#
# Usage
# -----
#   bash scripts/train_replication/train_smqr_seed2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SEED=2
ALGO_LABEL="smqr_cont_self_anchor_only"
RUN_TAG="exp_09_smqr_seed${SEED}"

run_training \
  --algo.config.critic-penalty-mode smqr_cont_self \
  --algo.config.sc-tau-res-scale 0.0
