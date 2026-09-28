#!/usr/bin/env bash
# Cross-motion table (LAFAN dance1): AW-CQL, seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
require_file "${LAFAN_H5}.aw_weights.npz" "run: python scripts/aw_precompute_weights.py ${LAFAN_H5}"
run_seeds g1_29dof_wbt_lafan_dance1_aw_cql g1-29dof-wbt-lafan-dance1-aw-cql
finish
