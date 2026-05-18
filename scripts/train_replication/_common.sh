#!/usr/bin/env bash
# Common helpers shared by replication train scripts (CQL/SMQR seeds 2,3).
#
# This file is *sourced* by the per-run scripts. It centralises:
#   * REPO_ROOT detection
#   * default env-var overrides (NUM_ITERS, SAVE_INTERVAL, LOGGER, PROJECT, EXP_ROOT)
#   * a post-train rename helper that aliases the auto-generated timestamped
#     directory to a stable name like  logs/hv-g1-manager/exp_09_<algo>_seed<N>
#
# All replication runs use the SAME offline dataset / preset / hyper-params as
# the seed=1 (exp_08) baseline; only `--training.seed` and `--training.name`
# differ between scripts.  No train-loss code is modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults (override via env vars on the calling line) ────────────────
: "${NUM_ITERS:=20000}"
: "${SAVE_INTERVAL:=5000}"
: "${LOGGER:=wandb}"          # logger:wandb (default) or logger:disabled
: "${PROJECT:=hv-g1-manager}"
: "${EXP_ROOT:=logs/${PROJECT}}"
: "${PRESET:=exp:g1-29dof-wbt-offline-cql-w-object}"
: "${PYTHON_BIN:=python}"
: "${RENAME_AFTER_TRAIN:=1}"  # 1 = mv timestamped dir to stable alias, 0 = leave as-is
: "${DRY_RUN:=0}"             # 1 = print command and exit (no training)
: "${DATASET_PATH:=offline_data/fastsac_dataset.h5}"  # offline H5 dataset (required by offline CQL)

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[replication] ERROR: DATASET_PATH not found: ${DATASET_PATH}" >&2
  echo "[replication]   Set DATASET_PATH=<abs-or-relative-path> before launching." >&2
  exit 3
fi

# Echo a header so background logs are easy to identify.
print_header () {
  echo "=================================================================="
  echo "[replication] tag=${RUN_TAG}"
  echo "[replication] seed=${SEED}  algo=${ALGO_LABEL}"
  echo "[replication] iters=${NUM_ITERS}  save_interval=${SAVE_INTERVAL}"
  echo "[replication] logger=${LOGGER}  project=${PROJECT}"
  echo "[replication] preset=${PRESET}"
  echo "[replication] target alias = ${EXP_ROOT}/${RUN_TAG}"
  echo "=================================================================="
}

# Find newest dir under EXP_ROOT whose name matches *-<RUN_TAG>-* (the
# experiment_paths.py format is  <timestamp>-<name>-<group_or_task>).
# Echo the absolute path or empty string if not found.
find_latest_run_dir () {
  local tag="$1"
  # shellcheck disable=SC2012
  ls -1dt "${EXP_ROOT}"/*-"${tag}"-* 2>/dev/null | head -n 1 || true
}

rename_to_stable_alias () {
  local stable="${EXP_ROOT}/${RUN_TAG}"
  if [[ -e "$stable" ]]; then
    echo "[replication] WARNING: alias '$stable' already exists; skipping rename."
    return 0
  fi
  local src
  src="$(find_latest_run_dir "${RUN_TAG}")"
  if [[ -z "$src" || ! -d "$src" ]]; then
    echo "[replication] WARNING: could not locate freshly-trained run dir for tag '${RUN_TAG}'."
    echo "[replication] You may need to mv it manually after inspecting ${EXP_ROOT}/."
    return 0
  fi
  echo "[replication] renaming '$src' -> '$stable'"
  mv "$src" "$stable"
}

run_training () {
  local extra_algo_args=("$@")

  local cmd=(
    "${PYTHON_BIN}" src/holosoma/holosoma/train_agent.py
    "${PRESET}"
    "logger:${LOGGER}"
    --training.seed "${SEED}"
    --training.name "${RUN_TAG}"
    --algo.config.dataset-path "${DATASET_PATH}"
    --algo.config.num-learning-iterations "${NUM_ITERS}"
    --algo.config.save-interval "${SAVE_INTERVAL}"
    "${extra_algo_args[@]}"
  )

  print_header
  echo "[replication] CMD: ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[replication] DRY_RUN=1, exiting before training."
    return 0
  fi

  "${cmd[@]}"

  if [[ "${RENAME_AFTER_TRAIN}" == "1" ]]; then
    rename_to_stable_alias
  fi
}
