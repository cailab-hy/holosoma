#!/usr/bin/env bash
# Cross-motion table (LAFAN kick (fightAndSports1_subject4 f1728-1858)): ACL-QL, seeds 1-5 (computes the ACL quality sidecar if missing).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
activate_env
ensure_acl_sidecar "${KICK_H5}"
run_seeds g1_29dof_wbt_lafan_kick_acl_ql g1-29dof-wbt-lafan-kick-acl-ql
finish
