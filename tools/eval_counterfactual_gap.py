"""Part B stub: replay syndiag counterfactual actions in the simulator (OUT OF SCOPE here).

Input contract
==============
This tool consumes the compressed dumps written by the BF-CQL synergy-OOD
diagnostics (Part A, ``BFCQLAgent._syndiag_dump``):

    {log_dir}/syndiag/dump_step{global_step:08d}.npz

Each dump holds one diagnostic batch (first ``dump_max_rows`` rows,
deterministic subsample). Keys (all numpy arrays, ``allow_pickle=False``;
N = rows, G = groups, C = coalitions, K = dump_topk, A = action dims):

    schema_version        int, currently 1
    global_step           int, critic global step of the dump
    dataset_path          str, source offline HDF5 dataset path
    group_names           [G]    str, BF-CQL action group names
    group_dim_mask        [G, A] bool, group -> action-dim membership
    coalition_names       [C]    str, e.g. "sing_LL", "pair_LL_RA", "tri_LL_RL_W"
    coalition_group_mask  [C, G] bool, coalition -> group membership
    coalition_dim_mask    [C, A] bool, coalition -> action-dim membership
    dataset_index         [N]    int64, global row in the source HDF5 dataset
                                 (-1 when unavailable, e.g. symmetry augmentation)
    observations_raw      [N, obs]  UNNORMALIZED actor observation as stored
    actions_raw           [N, A]    UNNORMALIZED (env-scale) dataset action as stored
    a_pi_norm / a_pi_env  [N, A]    deterministic actor action (normalized / env scale)
    drift                 [N, G]    per-group normalized-RMSE actor drift d_g
    v                     [N, C]    coalition value v(M) = minQ(s, a_cf(M)) - minQ(s, a_D)
    delta                 [N, C]    synergy residual Delta(M) = v(M) - sum_{g in M} v({g})
    q_data_min            [N]       min(Q1,Q2)(s, a_D) reused from the critic update
    q_cf                  [N, C]    min(Q1,Q2)(s, a_cf(M))
    top_coalition_ids     [N, K]    per-sample top coalitions by Delta
    top_delta             [N, K]    their Delta values
    a_cf_env_top          [N, K, A] env-scale counterfactual actions for those coalitions
    sigma                 [A]       running dataset-action std used by the drift metric

Row identity: ``dataset_index`` addresses the source HDF5 dataset directly, so
episode/motion metadata not present in the dump (``episode_id``,
``next_episode_step``, ``next_global_step``, ``motion_time_step``,
``motion_phase``, tracking errors, ...) can be looked up there.

Planned pipeline (Part B)
=========================
1. Load a dump and select the top-|Delta| (row, coalition) candidates.
2. For each candidate, reset the simulator to the stored transition state:
   locate the row via ``dataset_index`` in ``dataset_path`` and restore the
   motion command to the stored ``motion_time_step`` / episode phase (WBT
   supports deterministic resets to a motion timestep with pose overrides).
3. Execute the counterfactual action ``a_cf_env_top`` for the first control
   step, then continue with the deterministic policy for H steps
   (H ~ 10-50; sweep).
4. Record realized outcomes: fall / bad_tracking termination, tracking errors
   (err_root_pos, err_body_pos_max, ...), and the short-horizon discounted
   return under the training reward.
5. Compare against the critic's belief ``q_cf``: the counterfactual gap
   ``q_cf - realized_return_estimate``, binned by Delta and by block drift.
   The paper-facing figure plots overestimation gap vs Delta quartiles,
   mirroring the online ``syndiag/delta_pairs_driftQ{1..4}`` metrics.

This file is intentionally a stub; only the contract above is normative.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path, help="Path to a syndiag dump_step*.npz file")
    parser.add_argument("--horizon", type=int, default=20, help="Replay horizon H in control steps")
    parser.add_argument("--top-per-row", type=int, default=1, help="Counterfactual coalitions replayed per row")
    parser.parse_args()
    raise NotImplementedError(
        "Part B (simulator replay of syndiag counterfactuals) is not implemented yet. "
        "See the module docstring for the dump schema and the planned pipeline."
    )


if __name__ == "__main__":
    main()
