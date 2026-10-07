#!/usr/bin/env bash
# ODPR-CQL (agents/odpr_cql): CQL with ODPR-A (OPER-A) priority weights on the conservative bracket, i.e. the same
# weight placement as AW-CQL with a different weight signal. Weights: seeds 1-3 of scripts/oper_precompute_weights.py
# averaged, then ODPR's own load-time normalisation (linear rescale to std 2.0, floor 0.1). Seeds 1-5.
#   ODPR_MODE=ess  -> weights rescaled so ESS/N equals the AW H50 sidecar (sharpness-matched control)
#   ODPR_MODE=raw  -> unscaled OPER-A weights (nearly uniform on this dataset)
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
ODPR_MODE="${ODPR_MODE:-odpr}"
activate_env
case "${ODPR_MODE}" in
  odpr) SIDECAR="${G1_H5}.oper_a.odpr.npz"; TAG="odpr_cql" ;;
  ess)  SIDECAR="${G1_H5}.oper_a.ess_matched.npz"; TAG="odpr_cql_ess_matched" ;;
  raw)  SIDECAR="${G1_H5}.oper_a.raw.npz"; TAG="odpr_cql_raw" ;;
  *) echo "ODPR_MODE must be odpr, ess or raw"; exit 1 ;;
esac
ensure_oper_sidecar "${G1_H5}" "${ODPR_MODE}" "${SIDECAR}"
run_seeds "g1_29dof_wbt_${TAG}" g1-29dof-wbt-odpr-cql algo:odpr-cql --algo.config.odpr-weights-path "${SIDECAR}"
finish
