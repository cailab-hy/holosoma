#!/usr/bin/env bash
# Main G1-WBT table: BC, IQL, CQL (alpha=5), ACL-QL, AW-CQL (H50); seeds 1-5 each.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_bc.sh" "${D}/g1_wbt_iql.sh" "${D}/g1_wbt_cql.sh" "${D}/g1_wbt_acl_ql.sh" "${D}/g1_wbt_aw_cql.sh"
finish
