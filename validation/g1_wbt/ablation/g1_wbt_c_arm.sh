#!/usr/bin/env bash
# Ablation table: C-arm (formerly LSE-AW-CQL), w LSE - Q_D, seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
run_seeds g1_29dof_wbt_c_arm g1-29dof-wbt-c-arm
finish
