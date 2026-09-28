#!/usr/bin/env bash
# Structural ablation table (wBC, B-arm, C-arm, Asym-CQL); seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_wbc.sh" "${D}/g1_wbt_b_arm.sh" "${D}/g1_wbt_c_arm.sh" "${D}/g1_wbt_asym_cql.sh"
finish
