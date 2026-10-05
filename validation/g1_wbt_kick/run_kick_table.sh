#!/usr/bin/env bash
# Cross-motion (LAFAN kick (fightAndSports1_subject4 f1728-1858)) table: CQL, IQL, ACL-QL, AW-CQL; seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_kick_cql.sh" "${D}/g1_wbt_kick_iql.sh" "${D}/g1_wbt_kick_acl_ql.sh" "${D}/g1_wbt_kick_aw_cql.sh"
finish
