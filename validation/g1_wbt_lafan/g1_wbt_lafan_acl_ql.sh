#!/usr/bin/env bash
# Cross-motion table (LAFAN dance1): ACL-QL, seeds 1-3 (computes the ACL quality sidecar if missing).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3}"
activate_env
ensure_acl_sidecar "${LAFAN_H5}"
run_seeds g1_29dof_wbt_lafan_dance1_acl_ql g1-29dof-wbt-lafan-dance1-acl-ql
finish
