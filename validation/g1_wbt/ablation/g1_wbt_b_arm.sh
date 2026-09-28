#!/usr/bin/env bash
# Ablation table: B-arm (formerly OS-AW-CQL), LSE - w Q_D, seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
run_seeds g1_29dof_wbt_b_arm g1-29dof-wbt-b-arm
finish
