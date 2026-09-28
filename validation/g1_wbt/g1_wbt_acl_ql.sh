#!/usr/bin/env bash
# Main table: ACL-QL (Adam, raw weight outputs), seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
require_file "${G1_H5}.acl_quality.npz" "run: python scripts/acl_precompute_quality.py ${G1_H5}"
run_seeds g1_29dof_wbt_acl_ql g1-29dof-wbt-acl-ql
finish
