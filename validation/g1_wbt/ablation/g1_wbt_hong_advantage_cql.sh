#!/usr/bin/env bash
# Hong-Advantage CQL (agents/hong_advantage_cql): Hong et al. (ICLR 2023) trajectory Advantage Weighting as the
# minibatch sampling distribution, CQL objective unchanged (paired with g1_wbt_cql.sh). Seeds 1-5.
#   p(i,t) ~ exp(maxmin(G_i - V_lin(s_i0)) / HONG_TEMP), sampled with replacement; no sidecar, computed at setup.
#   HONG_TEMP=0.2 (paper default for CQL); other values get a _T<temp> run-name suffix.
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
HONG_TEMP="${HONG_TEMP:-0.2}"
activate_env
require_file "${G1_H5}" "collect the largebox dataset first"
TAG="g1_29dof_wbt_hong_advantage_cql"
[[ "${HONG_TEMP}" != "0.2" ]] && TAG="${TAG}_T${HONG_TEMP}"
run_seeds "${TAG}" g1-29dof-wbt-hong-advantage-cql algo:hong-advantage-cql --algo.config.hong-aw-temperature "${HONG_TEMP}"
finish
