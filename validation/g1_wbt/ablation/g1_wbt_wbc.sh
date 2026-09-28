#!/usr/bin/env bash
# Ablation table: weighted BC (wBC), seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
run_seeds g1_29dof_wbt_w_bc g1-29dof-wbt-w-bc
finish
