#!/usr/bin/env bash
# Main table: AW-CQL with the H=50 sidecar, seeds 1-5 (seeds 1-3 are reused as the H50 point of the H sweep).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
ensure_aw_sidecar "${G1_H5}" 50 "${G1_H5}.aw_weights.H50.npz"
run_seeds g1_29dof_wbt_aw_cql_H50 g1-29dof-wbt-aw-cql algo:aw-cql --algo.config.aw-weights-path "${G1_H5}.aw_weights.H50.npz"
finish
