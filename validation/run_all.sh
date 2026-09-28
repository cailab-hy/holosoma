#!/usr/bin/env bash
# Run every paper table: main (seeds 1-5), ablation, H sweep, LAFAN
# cross-motion (seeds 1-3). Optional scripts (TD3+BC, CQL alpha sweep) are not included.
# Finished seeds are skipped, so this can be re-run after an interruption.
#
# To use two GPUs/terminals without overlapping runs, give each terminal a disjoint part:
#   PART=1 bash validation/run_all.sh   # main table + H sweep        (31 runs)
#   PART=2 bash validation/run_all.sh   # ablation table + LAFAN table (33 runs)
# Without PART everything runs sequentially in one process.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
PART="${PART:-all}"
case "${PART}" in
  1) run_scripts "${D}/g1_wbt/run_main_table.sh" "${D}/g1_wbt/H_sweep/run_h_sweep.sh" ;;
  2) run_scripts "${D}/g1_wbt/ablation/run_ablation_table.sh" "${D}/g1_wbt_lafan/run_lafan_table.sh" ;;
  all)
    run_scripts \
      "${D}/g1_wbt/run_main_table.sh" \
      "${D}/g1_wbt/ablation/run_ablation_table.sh" \
      "${D}/g1_wbt/H_sweep/run_h_sweep.sh" \
      "${D}/g1_wbt_lafan/run_lafan_table.sh"
    ;;
  *) echo "PART must be 1, 2 or all (got '${PART}')"; exit 1 ;;
esac
finish
