#!/usr/bin/env bash
# H robustness (appendix): H25 and H100, seeds 1-3. H50 reuses g1_wbt_aw_cql.sh seeds 1-3.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
run_scripts "${D}/g1_wbt_h25.sh" "${D}/g1_wbt_h100.sh"
finish
