#!/usr/bin/env bash
# Cross-motion (LAFAN dance1) table: CQL, IQL, ACL-QL, AW-CQL; seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_lafan_cql.sh" "${D}/g1_wbt_lafan_iql.sh" "${D}/g1_wbt_lafan_acl_ql.sh" "${D}/g1_wbt_lafan_aw_cql.sh"
finish
