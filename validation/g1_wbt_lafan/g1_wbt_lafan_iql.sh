#!/usr/bin/env bash
# Cross-motion table (LAFAN dance1): IQL, seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
run_seeds g1_29dof_wbt_lafan_dance1_iql g1-29dof-wbt-lafan-dance1-iql
finish
