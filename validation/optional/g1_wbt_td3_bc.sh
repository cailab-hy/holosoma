#!/usr/bin/env bash
# Optional: TD3+BC, seeds 1-3 (set SEEDS="1 2 3 4 5" for a main-table run).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
run_seeds g1_29dof_wbt_td3_bc g1-29dof-wbt-td3-bc
finish
