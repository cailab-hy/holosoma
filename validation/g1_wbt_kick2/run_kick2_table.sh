#!/usr/bin/env bash
# Cross-motion (LAFAN double kick (fight1_subject3 f5000-5241, two events)) table: CQL, IQL, ACL-QL, AW-CQL; seeds 1-5.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_kick2_cql.sh" "${D}/g1_wbt_kick2_iql.sh" "${D}/g1_wbt_kick2_acl_ql.sh" "${D}/g1_wbt_kick2_aw_cql.sh"
finish
