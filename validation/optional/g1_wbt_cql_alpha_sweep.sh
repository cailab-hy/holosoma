#!/usr/bin/env bash
# Optional: CQL alpha sweep (alpha = 5 is the main-table reference and is not rerun here), seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
ALPHAS="${ALPHAS:-1.0 2.5 10.0}"
for alpha in ${ALPHAS}; do
  tag="alpha${alpha/./p}"
  run_seeds "g1_29dof_wbt_cql_${tag}" g1-29dof-wbt-cql algo:cql --algo.config.cql-weight "${alpha}"
done
finish
