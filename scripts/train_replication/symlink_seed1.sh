#!/usr/bin/env bash
# Replication run 3-E: alias the existing seed=1 baseline checkpoints
# (exp_08_cql / exp_08_smqr) under the seed-aware naming used by the new runs.
#
# This creates symlinks so the τ-probe / η-sweep aggregation scripts can treat
# all three seeds (1,2,3) uniformly via a single CKPT_DIR_FMT pattern.
#
# Idempotent: re-running will only create missing symlinks.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${EXP_ROOT:=logs/hv-g1-manager}"
: "${STEPS:=0005000 0010000 0015000 0020000}"

declare -A SRC_MAP=(
  [exp_09_cql_seed1]="${EXP_ROOT}/exp_08_cql"
  [exp_09_smqr_seed1]="${EXP_ROOT}/exp_08_smqr"
)

for tag in "${!SRC_MAP[@]}"; do
  src="${SRC_MAP[$tag]}"
  dst="${EXP_ROOT}/${tag}"
  if [[ ! -d "$src" ]]; then
    echo "[symlink_seed1] WARNING: source missing: $src  -- skipping ${tag}"
    continue
  fi
  mkdir -p "$dst"
  for step in $STEPS; do
    src_file="${src}/model_${step}.pt"
    dst_file="${dst}/model_${step}.pt"
    if [[ ! -f "$src_file" ]]; then
      echo "[symlink_seed1] note: $src_file not found, skipping."
      continue
    fi
    if [[ -L "$dst_file" || -f "$dst_file" ]]; then
      echo "[symlink_seed1] exists: $dst_file"
      continue
    fi
    ln -s "$(realpath "$src_file")" "$dst_file"
    echo "[symlink_seed1] linked $dst_file -> $src_file"
  done
done

echo "[symlink_seed1] done."
