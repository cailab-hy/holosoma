#!/usr/bin/env bash
# Frozen V_H baseline ablation: AW-CQL (PRe) with ONLY the advantage baseline swapped from the progress-bin mean
# b_k to a learned state-conditioned V_H(s) ~ E[G^H | s] (MSE regression on the same G^H, 5-fold cross-fitting by
# episode, frozen before CQL training). A = G^H - V_H(s); sigma, exp, w_max clip, unit mean, placement, alpha and
# seeds are identical to the main AW-CQL arm. See scripts/vh_precompute_weights.py.
#   VH_SIGMA=recomputed (default)  sigma_A recomputed from the V_H residual   -> g1_29dof_wbt_aw_cql_H50_vh_seed<N>
#   VH_SIGMA=pre                   PRe's sigma_A kept (same temperature)       -> g1_29dof_wbt_aw_cql_H50_vh_presigma_seed<N>
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
VH_SIGMA="${VH_SIGMA:-recomputed}"
activate_env
ensure_aw_sidecar "${G1_H5}" 50 "${G1_H5}.aw_weights.H50.npz"
ensure_vh_sidecar "${G1_H5}" 50
case "${VH_SIGMA}" in
  recomputed) SIDECAR="${G1_H5}.aw_weights.H50.vh.npz"; TAG="aw_cql_H50_vh" ;;
  pre)        SIDECAR="${G1_H5}.aw_weights.H50.vh_presigma.npz"; TAG="aw_cql_H50_vh_presigma" ;;
  *) echo "VH_SIGMA must be recomputed or pre"; exit 1 ;;
esac
run_seeds "g1_29dof_wbt_${TAG}" g1-29dof-wbt-aw-cql algo:aw-cql --algo.config.aw-weights-path "${SIDECAR}"
finish
