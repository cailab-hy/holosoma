#!/usr/bin/env bash
# H robustness (appendix): AW-CQL with the H=100 sidecar, seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
ensure_aw_sidecar "${G1_H5}" 100 "${G1_H5}.aw_weights.H100.npz"
run_seeds g1_29dof_wbt_aw_cql_H100 g1-29dof-wbt-aw-cql algo:aw-cql --algo.config.aw-weights-path "${G1_H5}.aw_weights.H100.npz"
finish
