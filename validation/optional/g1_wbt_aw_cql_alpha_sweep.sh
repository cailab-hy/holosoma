#!/usr/bin/env bash
# Optional: AW-CQL (H=50 sidecar) alpha sweep. alpha = 5 is the main-table reference and is not rerun here.
#   ALPHAS="1.0" SEEDS="1 2 3 4 5" bash validation/optional/g1_wbt_aw_cql_alpha_sweep.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
ALPHAS="${ALPHAS:-1.0 10.0}"
ensure_aw_sidecar "${G1_H5}" 50 "${G1_H5}.aw_weights.H50.npz"
for alpha in ${ALPHAS}; do
  tag="alpha${alpha/./p}"
  run_seeds "g1_29dof_wbt_aw_cql_H50_${tag}" g1-29dof-wbt-aw-cql algo:aw-cql \
    --algo.config.aw-weights-path "${G1_H5}.aw_weights.H50.npz" --algo.config.cql-weight "${alpha}"
done
finish
