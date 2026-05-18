#!/usr/bin/env bash
# Master orchestrator for the 4 replication training runs (seeds 2, 3 × CQL, SMQR).
#
# Sequential by default (one GPU assumed).  Set PARALLEL=1 to launch the four
# runs concurrently in the background — only do this if you actually have four
# GPUs free; otherwise they will OOM-fight on a single device.
#
# Usage
# -----
#   bash scripts/train_replication/run_all_seeds.sh                # sequential
#   DRY_RUN=1 bash scripts/train_replication/run_all_seeds.sh      # print only
#   ONLY="train_cql_seed2.sh train_smqr_seed2.sh" \
#       bash scripts/train_replication/run_all_seeds.sh            # subset
#
# Notes
# -----
# * Sequential mode logs each run to logs/train_replication/<tag>.log via tee.
# * The 4 runs are independent; failures are reported but do not abort siblings
#   when CONTINUE_ON_ERROR=1 (default 0 for sequential mode).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

: "${PARALLEL:=0}"
: "${CONTINUE_ON_ERROR:=0}"
: "${LOG_DIR:=logs/train_replication}"
: "${ONLY:=train_cql_seed2.sh train_cql_seed3.sh train_smqr_seed2.sh train_smqr_seed3.sh}"

mkdir -p "$LOG_DIR"

run_one () {
  local script="$1"
  local tag="${script%.sh}"
  local log="${LOG_DIR}/${tag}.log"
  echo "[run_all_seeds] >>> ${script}  (log: ${log})"
  if [[ "${PARALLEL}" == "1" ]]; then
    bash "${SCRIPT_DIR}/${script}" >"$log" 2>&1 &
  else
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "$log" || \
        echo "[run_all_seeds] WARNING: ${script} failed (CONTINUE_ON_ERROR=1)"
    else
      bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "$log"
    fi
  fi
}

for s in $ONLY; do
  if [[ ! -f "${SCRIPT_DIR}/${s}" ]]; then
    echo "[run_all_seeds] ERROR: missing ${SCRIPT_DIR}/${s}"
    exit 2
  fi
  run_one "$s"
done

if [[ "${PARALLEL}" == "1" ]]; then
  echo "[run_all_seeds] PARALLEL=1: waiting for background jobs ..."
  wait
fi

echo "[run_all_seeds] all requested runs completed."
echo "[run_all_seeds] next: bash scripts/train_replication/symlink_seed1.sh"
