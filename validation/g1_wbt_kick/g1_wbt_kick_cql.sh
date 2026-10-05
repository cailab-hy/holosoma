#!/usr/bin/env bash
# Cross-motion table (LAFAN kick (fightAndSports1_subject4 f1728-1858)): CQL, seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
run_seeds g1_29dof_wbt_lafan_kick_cql g1-29dof-wbt-lafan-kick-cql
finish
