#!/usr/bin/env bash
set -Eeuo pipefail

H5="${H5:-offline_data/g1_29dof_wbt_fastsac_episode1m_env256_dataset.h5}"
CACHE="${CACHE:-probe_rows_cell1_v3_full.npz}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
STEPS="${STEPS:-1k:end:1k}"
BINS=(0 2 4 5 6 7 11 12 13 14)

[[ -f "$H5" ]] || { echo "[error] dataset not found: $H5" >&2; exit 1; }
[[ -f "$CACHE" ]] || { echo "[error] index cache not found: $CACHE" >&2; exit 1; }

run_probe() {
  local label="$1"
  local checkpoint_dir="$2"
  local output="$3"
  [[ -d "$checkpoint_dir" ]] || {
    echo "[error] checkpoint directory not found: $checkpoint_dir" >&2
    exit 1
  }

  echo "[probe] label=$label output=$output"
  python aw_wall_probe.py "$H5" \
    --ckpt-dir "$checkpoint_dir" \
    --run-label "$label" \
    --algo cql \
    --steps "$STEPS" \
    --bins "${BINS[@]}" \
    --per-cell all \
    --span-n 3000000 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --index-cache "$CACHE" \
    --strict-index-cache \
    --out "$output"
}

run_probe \
  "cql-seed2" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260810_111347-g1_29dof_wbt_cql_manager_seed2-locomotion" \
  "cql-seed2-probe_v3_full.csv"

run_probe \
  "cql-seed1" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260717_120756-g1_29dof_wbt_cql_manager_pu1_4096-locomotion" \
  "cql-seed1-probe_v3_full.csv"

run_probe \
  "aw-cql-seed2" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260722_083418-g1_29dof_wbt_aw_cql_manager_pu1_4096_seed2-locomotion" \
  "aw-cql-seed2-probe_v3_full.csv"

run_probe \
  "aw-cql-seed1" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260717_162030-g1_29dof_wbt_aw_cql_manager_pu1_4096-locomotion" \
  "aw-cql-seed1-probe_v3_full.csv"

run_probe \
  "dw-cql-seed1" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260811_035318-g1_29dof_wbt_dw_cql_manager-locomotionn" \
  "dw-cql-seed1-probe_v3_full.csv"

run_probe \
  "lse-aw-cql-seed1-C" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260812_124923-g1_29dof_wbt_lse_aw_cql_manager_seed1-locomotion" \
  "dw-cql-seed1-C-probe_v3_full.csv"

run_probe \
  "osw-cql-seed1-B" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260820_043018-g1_29dof_wbt_os_aw_cql_manager_seed1-locomotion" \
  "os-aw-cql-seed1-B-probe_v3_full.csv"

run_probe \
  "cql-weight1-seed1" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260818_115927-g1_29dof_wbt_cql_manager_weight1_4096_300k-locomotion" \
  "cql-weight1-seed1-probe_v3_full.csv"

run_probe \
  "aw-cql-weight1-seed1" \
  "/home/cai/holosoma/logs/WholeBodyTracking/20260819_035805-g1_29dof_wbt_aw_cql_manager_weight1_4096-locomotion" \
  "aw-cql-weight1-seed1-probe_v3_full.csv"