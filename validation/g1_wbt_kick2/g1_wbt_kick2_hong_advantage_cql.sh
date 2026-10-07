#!/usr/bin/env bash
# Cross-motion table (LAFAN double kick): Hong-Advantage CQL (Hong et al. ICLR 2023 trajectory AW sampling,
# temperature 0.2, CQL objective unchanged), seeds 1-5. Paired with g1_wbt_kick2_cql.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
require_file "${KICK2_H5}" "run g1_wbt_kick2_collect.sh first"
run_seeds g1_29dof_wbt_lafan_kick2_hong_advantage_cql g1-29dof-wbt-lafan-kick2-hong-advantage-cql
finish
