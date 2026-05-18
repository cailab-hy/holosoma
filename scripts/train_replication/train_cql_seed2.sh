#!/usr/bin/env bash
# Replication run 3-A: vanilla CQL, seed=2, 20k iters.
#
# Same offline dataset & preset as the seed=1 baseline (exp_08_cql).
# Output directory will be renamed to  logs/hv-g1-manager/exp_09_cql_seed2/
# (override via EXP_ROOT / RENAME_AFTER_TRAIN).
#
# Usage
# -----
#   bash scripts/train_replication/train_cql_seed2.sh
#   DRY_RUN=1   bash scripts/train_replication/train_cql_seed2.sh   # print only
#   LOGGER=disabled bash scripts/train_replication/train_cql_seed2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SEED=2
ALGO_LABEL="vanilla_cql"
RUN_TAG="exp_09_cql_seed${SEED}"

run_training \
  --algo.config.critic-penalty-mode vanilla_cql
