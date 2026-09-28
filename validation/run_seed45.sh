#!/usr/bin/env bash
# Bring the remaining paper arms from 3 to 5 seeds.
# Two disjoint parts for two terminals; finished seeds are skipped automatically.
#   PART=1 bash validation/run_seed45.sh   # H25/H100 seeds 4-5, LAFAN CQL/IQL seeds 4-5      (8 runs)
#   PART=2 bash validation/run_seed45.sh   # LAFAN ACL-QL/AW-CQL seeds 4-5                    (4 runs)
#   OPTIONAL=1 PART=2 ...                  # also wBC / C-arm / Asym-CQL seeds 4-5 (collapsed arms, +6 runs)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
PART="${PART:-1}"
case "${PART}" in
  1)
    SEEDS="4 5" bash "${D}/g1_wbt/H_sweep/g1_wbt_h25.sh"
    SEEDS="4 5" bash "${D}/g1_wbt/H_sweep/g1_wbt_h100.sh"
    SEEDS="4 5" bash "${D}/g1_wbt_lafan/g1_wbt_lafan_cql.sh"
    SEEDS="4 5" bash "${D}/g1_wbt_lafan/g1_wbt_lafan_iql.sh"
    ;;
  2)
    SEEDS="4 5" bash "${D}/g1_wbt_lafan/g1_wbt_lafan_acl_ql.sh"
    SEEDS="4 5" bash "${D}/g1_wbt_lafan/g1_wbt_lafan_aw_cql.sh"
    if [[ "${OPTIONAL:-0}" == "1" ]]; then
      SEEDS="4 5" bash "${D}/g1_wbt/ablation/g1_wbt_wbc.sh"
      SEEDS="4 5" bash "${D}/g1_wbt/ablation/g1_wbt_c_arm.sh"
      SEEDS="4 5" bash "${D}/g1_wbt/ablation/g1_wbt_asym_cql.sh"
    fi
    ;;
  *) echo "PART must be 1 or 2 (got '${PART}')"; exit 1 ;;
esac
finish
