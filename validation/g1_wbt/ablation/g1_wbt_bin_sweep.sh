#!/usr/bin/env bash
# Progress-bin sensitivity of the phase-conditioned advantage baseline b(kappa): AW-CQL H=50 with the
# per-bin baseline computed on K progress bins (main arm: K=20; global baseline control: K=1).
# Everything else (H=50, beta = sigma(A) re-estimated per K, clipping, normalisation, training) is identical.
#   KS="10 40" SEEDS="1 2 3 4 5" bash validation/g1_wbt/ablation/g1_wbt_bin_sweep.sh
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
SEEDS="${SEEDS:-1 2 3 4 5}"
KS="${KS:-10 40}"
activate_env
for K in ${KS}; do
  SIDECAR="${G1_H5}.aw_weights.H50.K${K}.npz"
  ensure_aw_sidecar_bins "${G1_H5}" 50 "${K}" "${SIDECAR}"
  run_seeds "g1_29dof_wbt_aw_cql_H50_K${K}" g1-29dof-wbt-aw-cql algo:aw-cql \
    --algo.config.aw-weights-path "${SIDECAR}"
done
finish
