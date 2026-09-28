#!/usr/bin/env bash
# Ablation table: Asym-CQL (independent +/- advantage weights), seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
require_file "${G1_H5}.asym_weights.npz" "asym sidecar is required (see scripts/asym_precompute_weights.py)"
run_seeds g1_29dof_wbt_asym_cql g1-29dof-wbt-asym-cql
finish
