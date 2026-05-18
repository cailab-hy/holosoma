#!/usr/bin/env bash
# Stage pre-existing eval results into the Step-1 layout via symlinks.
set -e
cd "$(dirname "$0")/../.."
declare -A MAP=(
  ["eval_out/step1/cql_golden_300k/step_300000"]="eval_out/perf_stage1_cql/step_300000"
  ["eval_out/step1/smqr_anchor_golden_300k/step_300000"]="eval_out/perf_stage1_smqr_qtg/step_300000"
  ["eval_out/step1/smqr_sg_current_best_1m/step_1000000"]="eval_out/s6_1m/step_1000000"
  ["eval_out/step1/smqr_sg_sweep_default/step_300000"]="eval_out/perf_stage1_smqrsg/step_300000"
  ["eval_out/step1/smqr_sg_sweep_default/step_500000"]="eval_out/s6_1m/step_500000"
  ["eval_out/step1/smqr_sg_sweep_default/step_750000"]="eval_out/s6_1m/step_750000"
  ["eval_out/step1/smqr_sg_sweep_default/step_1000000"]="eval_out/s6_1m/step_1000000"
)
for dst in "${!MAP[@]}"; do
  src="${MAP[$dst]}"
  if [[ ! -f "$src/eval_summary.json" ]]; then
    echo "[skip] $src/eval_summary.json missing"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  [[ -L "$dst" || -e "$dst" ]] && rm -rf "$dst"
  ln -s "$PWD/$src" "$dst"
  echo "[ok] $dst -> $src"
done
