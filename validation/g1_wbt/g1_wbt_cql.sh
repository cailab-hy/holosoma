#!/usr/bin/env bash
# Main table: CQL reference (alpha = 5, set in the experiment config), seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
run_seeds g1_29dof_wbt_cql g1-29dof-wbt-cql
finish
