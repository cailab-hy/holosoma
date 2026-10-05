#!/usr/bin/env bash
# Cross-motion table (LAFAN kick (fightAndSports1_subject4 f1728-1858)): AW-CQL, seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
require_file "${KICK_H5}.aw_weights.npz" "run: python scripts/aw_precompute_weights.py ${KICK_H5}"
run_seeds g1_29dof_wbt_lafan_kick_aw_cql g1-29dof-wbt-lafan-kick-aw-cql
finish
