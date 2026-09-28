#!/usr/bin/env bash
# LAFAN dance1 counterpart of validation/g1_wbt/ablation/g1_wbt_aw_cql_global_baseline.sh:
# AW-CQL trained with weights whose H-step advantage uses ONE dataset-wide baseline instead of the
# per-phase-bin baseline. Everything else (H=50, beta rule, clipping, normalisation, AW-CQL training)
# is identical to the LAFAN AW-CQL main arm.
#   SEEDS="1 2 3" bash validation/g1_wbt_lafan/g1_wbt_lafan_aw_cql_global_baseline.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
ensure_aw_sidecar_global "${LAFAN_H5}" 50 "${LAFAN_H5}.aw_weights.H50.global.npz"
run_seeds g1_29dof_wbt_lafan_dance1_aw_cql_H50_global g1-29dof-wbt-lafan-dance1-aw-cql algo:aw-cql \
  --algo.config.aw-weights-path "${LAFAN_H5}.aw_weights.H50.global.npz"
finish
