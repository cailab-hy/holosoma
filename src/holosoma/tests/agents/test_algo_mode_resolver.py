"""Unit tests for the Phase A unified algo-mode resolver.

These tests run *without* torch / pydantic to keep them lightweight.
They use a ``SimpleNamespace`` as a stand-in for ``OfflineCQLConfig``
since the resolver only reads attributes via ``getattr``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from holosoma.agents.offline_rl.common.algo_mode import (
    LEGACY_SMQR,
    LEGACY_VANILLA_CQL,
    MODE_AUTO,
    MODE_CQL,
    MODE_SMQR_ANCHOR,
    MODE_SMQR_LEARNED,
    TAU_ANCHOR,
    TAU_LEARNED,
    TAU_NONE,
    assert_phase_a_compatible,
    format_run_name,
    resolve_algo_mode,
)


def _ns(**kwargs):
    """Build a minimal namespace with only the keys we care about."""
    defaults = dict(
        algo_mode=MODE_AUTO,
        critic_penalty_mode=LEGACY_VANILLA_CQL,
        sc_tau_res_scale=0.0,
        smqr_learned_variant="vanilla",
        smqr_logging_namespace=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── Auto / legacy inference ────────────────────────────────────────


def test_auto_legacy_vanilla_cql_resolves_to_cql():
    args = _ns(critic_penalty_mode=LEGACY_VANILLA_CQL)
    r = resolve_algo_mode(args)
    assert r.mode == MODE_CQL
    assert r.tau_source == TAU_NONE
    assert r.logging_prefix == f"train/{MODE_CQL}/"
    assert r.run_name_tag == "cql"
    assert r.explicit is False


def test_auto_legacy_smqr_anchor_zero_scale_resolves_to_anchor():
    args = _ns(critic_penalty_mode=LEGACY_SMQR, sc_tau_res_scale=0.0)
    r = resolve_algo_mode(args)
    assert r.mode == MODE_SMQR_ANCHOR
    assert r.tau_source == TAU_ANCHOR
    assert r.logging_prefix == f"train/{MODE_SMQR_ANCHOR}/"
    assert r.run_name_tag == "smqranc"


def test_auto_legacy_smqr_positive_scale_resolves_to_learned():
    # Even in 'auto' the resolver labels this as smqr_learned — but
    # downstream the Phase A guard will block training.
    args = _ns(critic_penalty_mode=LEGACY_SMQR, sc_tau_res_scale=2.0)
    r = resolve_algo_mode(args)
    assert r.mode == MODE_SMQR_LEARNED
    assert r.tau_source == TAU_LEARNED


# ── Explicit modes — happy paths ───────────────────────────────────


def test_explicit_cql_with_consistent_legacy():
    args = _ns(algo_mode=MODE_CQL, critic_penalty_mode=LEGACY_VANILLA_CQL)
    r = resolve_algo_mode(args)
    assert r.mode == MODE_CQL
    assert r.explicit is True


def test_explicit_smqr_anchor_with_consistent_legacy():
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    r = resolve_algo_mode(args)
    assert r.mode == MODE_SMQR_ANCHOR
    assert r.explicit is True


def test_explicit_smqr_learned_with_consistent_legacy_resolves_but_guard_blocks():
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    )
    r = resolve_algo_mode(args)
    assert r.mode == MODE_SMQR_LEARNED
    # Guard must fire for the learned branch in Phase A.
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r)


# ── Explicit modes — conflicts ─────────────────────────────────────


def test_explicit_cql_conflicts_with_smqr_legacy():
    args = _ns(algo_mode=MODE_CQL, critic_penalty_mode=LEGACY_SMQR)
    with pytest.raises(ValueError, match="algo_mode='cql'"):
        resolve_algo_mode(args)


def test_explicit_smqr_anchor_conflicts_with_nonzero_scale():
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    )
    with pytest.raises(ValueError, match="sc_tau_res_scale=0"):
        resolve_algo_mode(args)


def test_explicit_smqr_learned_conflicts_with_zero_scale():
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    with pytest.raises(ValueError, match="sc_tau_res_scale>0"):
        resolve_algo_mode(args)


# ── Variant placeholder ────────────────────────────────────────────


def test_stabilized_variant_blocked_by_phase_a_guard():
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,  # not even learned — guard still trips
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
        smqr_learned_variant="stabilized",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="stabilized"):
        assert_phase_a_compatible(r)


def test_unknown_variant_rejected():
    args = _ns(smqr_learned_variant="something_new")
    with pytest.raises(ValueError, match="smqr_learned_variant"):
        resolve_algo_mode(args)


def test_unknown_mode_rejected():
    args = _ns(algo_mode="bogus")
    with pytest.raises(ValueError, match="algo_mode='bogus'"):
        resolve_algo_mode(args)


# ── Logging namespace override + run name ──────────────────────────


def test_logging_namespace_override_used_verbatim():
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
        smqr_logging_namespace="custom/ns",
    )
    r = resolve_algo_mode(args)
    assert r.logging_prefix == "custom/ns/"


def test_format_run_name_includes_mode_and_seed():
    args = _ns(critic_penalty_mode=LEGACY_VANILLA_CQL)
    r = resolve_algo_mode(args)
    name = format_run_name("exp_14_baseline", r, seed=7)
    assert name == "exp_14_baseline__mode-cql__seed7"


# ── Backward compat — default ctor mirrors legacy ──────────────────


def test_default_namespace_equivalent_to_legacy_vanilla_cql():
    """Empty namespace (only required defaults) must resolve to cql/none."""
    args = _ns()
    r = resolve_algo_mode(args)
    assert r.mode == MODE_CQL
    assert r.tau_source == TAU_NONE
    assert r.legacy_critic_penalty_mode == LEGACY_VANILLA_CQL
    # Phase A guard must accept the default path.
    assert_phase_a_compatible(r)


# ── Phase B opt-in gate ────────────────────────────────────────────


def test_phase_b_optin_allows_learned_vanilla():
    """Opt-in flag must open the gate for the vanilla learned branch."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    )
    r = resolve_algo_mode(args)
    # No raise.
    assert_phase_a_compatible(r, allow_learned=True)
    assert r.mode == MODE_SMQR_LEARNED
    assert r.tau_source == TAU_LEARNED


def test_phase_b_optin_does_not_unlock_stabilized():
    """Even with opt-in, the stabilized variant must remain gated."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="stabilized",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="stabilized"):
        assert_phase_a_compatible(r, allow_learned=True)


def test_phase_b_optin_does_not_affect_anchor_path():
    """Opt-in must not change anchor-only / cql semantics."""
    for cfg in (
        _ns(),  # default cql
        _ns(  # anchor-only
            algo_mode=MODE_SMQR_ANCHOR,
            critic_penalty_mode=LEGACY_SMQR,
            sc_tau_res_scale=0.0,
        ),
    ):
        r = resolve_algo_mode(cfg)
        # Both variants of the gate must accept these paths.
        assert_phase_a_compatible(r, allow_learned=False)
        assert_phase_a_compatible(r, allow_learned=True)


def test_phase_a_default_still_blocks_learned_without_optin():
    """Regression: default ``allow_learned=False`` must keep blocking."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r)  # no opt-in


# ── Phase C opt-in gate (stabilized variant) ──────────────────────


def test_phase_c_optin_unlocks_stabilized():
    """Both opt-ins together must permit the stabilized learned-τ branch."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="stabilized",
    )
    r = resolve_algo_mode(args)
    # No raise.
    assert_phase_a_compatible(r, allow_learned=True, allow_stabilized=True)
    assert r.mode == MODE_SMQR_LEARNED
    assert r.learned_variant == "stabilized"


def test_phase_c_optin_alone_does_not_unlock():
    """Phase C opt-in without Phase B must still be blocked."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="stabilized",
    )
    r = resolve_algo_mode(args)
    # Without allow_learned, the smqr_learned gate fires first.
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r, allow_learned=False, allow_stabilized=True)


def test_phase_c_optin_does_not_affect_vanilla_learned_path():
    """Phase C opt-in must not perturb the vanilla learned-τ gate."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="vanilla",
    )
    r = resolve_algo_mode(args)
    # vanilla + Phase B + Phase C (irrelevant) must pass.
    assert_phase_a_compatible(r, allow_learned=True, allow_stabilized=True)
    # vanilla + Phase B alone must still pass.
    assert_phase_a_compatible(r, allow_learned=True, allow_stabilized=False)
    # vanilla without Phase B must still fail (no opt-in laundering).
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r, allow_learned=False, allow_stabilized=True)


def test_phase_c_stabilized_invalid_for_non_learned_modes():
    """Stabilized variant on cql / smqr_anchor must be rejected even with opt-in."""
    for cfg in (
        _ns(smqr_learned_variant="stabilized"),  # cql + stabilized
        _ns(  # anchor + stabilized
            algo_mode=MODE_SMQR_ANCHOR,
            critic_penalty_mode=LEGACY_SMQR,
            sc_tau_res_scale=0.0,
            smqr_learned_variant="stabilized",
        ),
    ):
        r = resolve_algo_mode(cfg)
        with pytest.raises(NotImplementedError, match="stabilized"):
            assert_phase_a_compatible(
                r, allow_learned=True, allow_stabilized=True,
            )


# ── Phase D opt-in gate (V1 oneside_shrink variant) ───────────────


def test_phase_d_optin_unlocks_v1_oneside_shrink():
    """Both Phase B and Phase D opt-ins together must permit V1."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="v1_oneside_shrink",
    )
    r = resolve_algo_mode(args)
    # No raise.
    assert_phase_a_compatible(r, allow_learned=True, allow_v1=True)
    assert r.mode == MODE_SMQR_LEARNED
    assert r.learned_variant == "v1_oneside_shrink"


def test_phase_d_optin_alone_does_not_unlock():
    """Phase D opt-in without Phase B must still be blocked (smqr_learned gate)."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="v1_oneside_shrink",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r, allow_learned=False, allow_v1=True)


def test_v1_unaffected_by_stabilized_optin():
    """allow_stabilized must NOT silently unlock V1 (and vice-versa)."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="v1_oneside_shrink",
    )
    r = resolve_algo_mode(args)
    # B + C only (no D) → V1 must still be blocked.
    with pytest.raises(NotImplementedError, match="v1_oneside_shrink"):
        assert_phase_a_compatible(
            r, allow_learned=True, allow_stabilized=True, allow_v1=False,
        )

    # Conversely, stabilized must NOT be unlocked by allow_v1 alone.
    args_stab = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="stabilized",
    )
    r_stab = resolve_algo_mode(args_stab)
    with pytest.raises(NotImplementedError, match="stabilized"):
        assert_phase_a_compatible(
            r_stab, allow_learned=True, allow_stabilized=False, allow_v1=True,
        )


def test_v1_invalid_for_non_learned_modes():
    """V1 variant on cql / smqr_anchor must be rejected even with full opt-ins."""
    for cfg in (
        _ns(smqr_learned_variant="v1_oneside_shrink"),  # cql + v1
        _ns(  # anchor + v1
            algo_mode=MODE_SMQR_ANCHOR,
            critic_penalty_mode=LEGACY_SMQR,
            sc_tau_res_scale=0.0,
            smqr_learned_variant="v1_oneside_shrink",
        ),
    ):
        r = resolve_algo_mode(cfg)
        with pytest.raises(NotImplementedError, match="v1_oneside_shrink"):
            assert_phase_a_compatible(
                r, allow_learned=True, allow_stabilized=True, allow_v1=True,
            )


# ── Phase E opt-in gate (anchor + stabilised objective) ───────────


def test_phase_e_default_anchor_vanilla_passes():
    """Anchor-only with default ``smqr_anchor_objective='vanilla'``
    must remain bit-equivalent (no opt-in needed)."""
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    r = resolve_algo_mode(args)
    # Default kwargs (anchor_objective='vanilla') must accept.
    assert_phase_a_compatible(r)
    # Explicit vanilla also accepted, with or without opt-in.
    assert_phase_a_compatible(r, anchor_objective="vanilla")
    assert_phase_a_compatible(
        r, anchor_objective="vanilla", allow_anchor_stab=True,
    )


def test_phase_e_optin_required_for_anchor_stabilized():
    """Anchor + stabilised objective must be gated until Phase E opt-in."""
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="Phase E opt-in"):
        assert_phase_a_compatible(r, anchor_objective="stabilized")


def test_phase_e_optin_unlocks_anchor_stabilized():
    """Phase E opt-in opens the gate for anchor + stabilised objective."""
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    r = resolve_algo_mode(args)
    assert_phase_a_compatible(
        r, anchor_objective="stabilized", allow_anchor_stab=True,
    )


def test_phase_e_anchor_stab_invalid_for_non_anchor_modes():
    """anchor_objective='stabilized' must be rejected for cql / smqr_learned
    even with full opt-ins."""
    # cql + anchor_stab
    r_cql = resolve_algo_mode(_ns())
    with pytest.raises(NotImplementedError, match="only valid with .*smqr_anchor"):
        assert_phase_a_compatible(
            r_cql, anchor_objective="stabilized", allow_anchor_stab=True,
        )
    # smqr_learned + anchor_stab
    r_lrn = resolve_algo_mode(_ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    ))
    with pytest.raises(NotImplementedError, match="only valid with .*smqr_anchor"):
        assert_phase_a_compatible(
            r_lrn,
            allow_learned=True,
            anchor_objective="stabilized",
            allow_anchor_stab=True,
        )


def test_phase_e_unknown_anchor_objective_rejected():
    """Unknown anchor_objective string must raise ValueError."""
    args = _ns(
        algo_mode=MODE_SMQR_ANCHOR,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=0.0,
    )
    r = resolve_algo_mode(args)
    with pytest.raises(ValueError, match="smqr_anchor_objective"):
        assert_phase_a_compatible(r, anchor_objective="something_new")


def test_phase_e_optin_does_not_affect_other_paths():
    """Phase E opt-in must NOT silently unlock learned-τ paths."""
    # learned + anchor_stab opt-in → learned must still need its own opt-in.
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(
            r, allow_anchor_stab=True,  # irrelevant for learned mode
        )


# ── Phase F opt-in gate (f1_st_qg variant) ────────────────────────


def test_phase_f_optin_unlocks_f1_st_qg():
    """Both Phase B and Phase F opt-ins together must permit F1."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
    )
    r = resolve_algo_mode(args)
    # No raise.
    assert_phase_a_compatible(r, allow_learned=True, allow_f1=True)
    assert r.mode == MODE_SMQR_LEARNED
    assert r.learned_variant == "f1_st_qg"


def test_phase_f_optin_alone_does_not_unlock():
    """Phase F opt-in without Phase B must still be blocked (smqr_learned gate)."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="smqr_learned"):
        assert_phase_a_compatible(r, allow_learned=False, allow_f1=True)


def test_phase_f_b_optin_alone_does_not_unlock_f1():
    """Phase B opt-in alone (without F) must keep F1 gated."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="f1_st_qg"):
        assert_phase_a_compatible(r, allow_learned=True, allow_f1=False)


def test_f1_unaffected_by_other_optins():
    """allow_stabilized / allow_v1 must NOT silently unlock F1 (and vice-versa)."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="f1_st_qg"):
        assert_phase_a_compatible(
            r,
            allow_learned=True, allow_stabilized=True, allow_v1=True,
            allow_f1=False,
        )
    # Conversely, V1 must NOT be unlocked by allow_f1 alone.
    args_v1 = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="v1_oneside_shrink",
    )
    r_v1 = resolve_algo_mode(args_v1)
    with pytest.raises(NotImplementedError, match="v1_oneside_shrink"):
        assert_phase_a_compatible(
            r_v1, allow_learned=True, allow_v1=False, allow_f1=True,
        )


def test_f1_invalid_for_non_learned_modes():
    """F1 variant on cql / smqr_anchor must be rejected even with full opt-ins."""
    for cfg in (
        _ns(smqr_learned_variant="f1_st_qg"),  # cql + f1
        _ns(  # anchor + f1
            algo_mode=MODE_SMQR_ANCHOR,
            critic_penalty_mode=LEGACY_SMQR,
            sc_tau_res_scale=0.0,
            smqr_learned_variant="f1_st_qg",
        ),
    ):
        r = resolve_algo_mode(cfg)
        with pytest.raises(NotImplementedError, match="f1_st_qg"):
            assert_phase_a_compatible(
                r,
                allow_learned=True, allow_stabilized=True,
                allow_v1=True, allow_f1=True,
            )


# ── Phase G1 (sub-flag of F1) — semantic neutrality ───────────────
# G1 (`smqr_f1_random_full_grad`) is a runtime-only sub-flag with no
# resolver semantics.  It must NOT change mode resolution and must NOT
# unlock any gated branch on its own.


def test_g1_subflag_does_not_alter_resolver():
    """smqr_f1_random_full_grad on/off must produce identical resolution."""
    base = dict(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
    )
    r_off = resolve_algo_mode(_ns(**base, smqr_f1_random_full_grad=False))
    r_on = resolve_algo_mode(_ns(**base, smqr_f1_random_full_grad=True))
    assert r_off.mode == r_on.mode == MODE_SMQR_LEARNED
    assert r_off.tau_source == r_on.tau_source == TAU_LEARNED
    assert r_off.learned_variant == r_on.learned_variant == "f1_st_qg"


def test_g1_subflag_does_not_unlock_f1_gate():
    """G1 sub-flag must NOT bypass the Phase F opt-in gate."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
        smqr_f1_random_full_grad=True,  # G1 set, but F gate closed
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="f1_st_qg"):
        assert_phase_a_compatible(r, allow_learned=True, allow_f1=False)


# ── Phase H1 (sub-flag of G1) — semantic neutrality ───────────────
# H1 (`smqr_h1_alpha_floor`) is a runtime-only float sub-flag with no
# resolver semantics.  It must NOT alter mode resolution and must NOT
# unlock any gated branch on its own.


def test_h1_subflag_does_not_alter_resolver():
    """smqr_h1_alpha_floor must produce identical resolution for any α."""
    base = dict(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
        smqr_f1_random_full_grad=True,
    )
    r_off = resolve_algo_mode(_ns(**base, smqr_h1_alpha_floor=0.0))
    r_on = resolve_algo_mode(_ns(**base, smqr_h1_alpha_floor=0.05))
    assert r_off.mode == r_on.mode == MODE_SMQR_LEARNED
    assert r_off.tau_source == r_on.tau_source == TAU_LEARNED
    assert r_off.learned_variant == r_on.learned_variant == "f1_st_qg"


def test_h1_subflag_does_not_unlock_f1_gate():
    """H1 sub-flag must NOT bypass the Phase F opt-in gate."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
        smqr_f1_random_full_grad=True,
        smqr_h1_alpha_floor=0.05,  # H1 set, but F gate closed
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="f1_st_qg"):
        assert_phase_a_compatible(r, allow_learned=True, allow_f1=False)


# ── Phase B2 (sub-flag of G1) — semantic neutrality ───────────────
# B2 (`smqr_b2_alpha_floor`) is a runtime-only float sub-flag with no
# resolver semantics.  It must NOT alter mode resolution, must NOT
# unlock any gated branch on its own, and at α=0 must produce a
# bit-exact regression to G1 (forward + Q-grad + τ-grad).


def test_b2_subflag_does_not_alter_resolver():
    """smqr_b2_alpha_floor must produce identical resolution for any α."""
    base = dict(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
        smqr_f1_random_full_grad=True,
    )
    r_off = resolve_algo_mode(_ns(**base, smqr_b2_alpha_floor=0.0))
    r_on = resolve_algo_mode(_ns(**base, smqr_b2_alpha_floor=0.05))
    assert r_off.mode == r_on.mode == MODE_SMQR_LEARNED
    assert r_off.tau_source == r_on.tau_source == TAU_LEARNED
    assert r_off.learned_variant == r_on.learned_variant == "f1_st_qg"


def test_b2_subflag_does_not_unlock_f1_gate():
    """B2 sub-flag must NOT bypass the Phase F opt-in gate."""
    args = _ns(
        algo_mode=MODE_SMQR_LEARNED,
        critic_penalty_mode=LEGACY_SMQR,
        sc_tau_res_scale=2.0,
        smqr_learned_variant="f1_st_qg",
        smqr_f1_random_full_grad=True,
        smqr_b2_alpha_floor=0.05,  # B2 set, but F gate closed
    )
    r = resolve_algo_mode(args)
    with pytest.raises(NotImplementedError, match="f1_st_qg"):
        assert_phase_a_compatible(r, allow_learned=True, allow_f1=False)


def test_b2_alpha_zero_bit_exact_to_g1():
    """B2 with α=0 must reproduce G1 bit-exactly: forward, Q-grad, τ-grad.

    This is the α=0 regression sanity required by the Phase I memo.
    Construct identical (Q, g) tensors, compute G1's vanilla `Q·g` and
    B2's STE form with α=0, and verify all three gradient channels
    match to float precision.
    """
    import torch

    torch.manual_seed(0)
    Q = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    # `g` must depend on a τ-like parameter to test τ-gradients.
    tau = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    g = torch.sigmoid(tau)  # g ∈ (0, 1), differentiable in τ

    # ── G1: vanilla Q·g ──
    qg_g1 = Q * g
    grad_Q_g1 = torch.autograd.grad(qg_g1.sum(), Q, retain_graph=True)[0]
    grad_tau_g1 = torch.autograd.grad(qg_g1.sum(), tau, retain_graph=True)[0]

    # ── B2 with α = 0 ──
    alpha = 0.0
    g_back = torch.clamp(g.detach(), min=alpha)
    qg_b2 = Q * g_back + Q.detach() * g - Q.detach() * g_back
    grad_Q_b2 = torch.autograd.grad(qg_b2.sum(), Q, retain_graph=True)[0]
    grad_tau_b2 = torch.autograd.grad(qg_b2.sum(), tau, retain_graph=True)[0]

    # Forward bit-exact
    assert torch.allclose(qg_b2, qg_g1, atol=0.0, rtol=0.0), (
        "B2(α=0) forward must be bit-exact to G1 forward"
    )
    # Q-gradient bit-exact
    assert torch.allclose(grad_Q_b2, grad_Q_g1, atol=1e-12, rtol=0.0), (
        "B2(α=0) Q-grad must be bit-exact to G1 (= g)"
    )
    # τ-gradient bit-exact
    assert torch.allclose(grad_tau_b2, grad_tau_g1, atol=1e-12, rtol=0.0), (
        "B2(α=0) τ-grad must be bit-exact to G1 (= Q·g')"
    )


def test_b2_alpha_positive_floors_q_grad_and_preserves_forward():
    """B2 with α>0 must preserve forward = Q·g and floor Q-grad ≥ α.

    τ-grad must remain Q·g' (= G1), independent of α.
    """
    import torch

    torch.manual_seed(0)
    Q = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    tau = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    # Force a fraction of the gates below the floor by squashing tau.
    g = torch.sigmoid(tau) * 0.1  # g ∈ (0, 0.1), most below α=0.05

    alpha = 0.05
    # G1 reference
    qg_g1 = Q * g
    grad_tau_g1 = torch.autograd.grad(qg_g1.sum(), tau, retain_graph=True)[0]

    # B2
    g_back = torch.clamp(g.detach(), min=alpha)
    qg_b2 = Q * g_back + Q.detach() * g - Q.detach() * g_back

    # Forward identity (Q·g)
    assert torch.allclose(qg_b2, Q * g, atol=1e-12), (
        "B2 forward must equal Q·g exactly"
    )
    # Q-grad equals max(g, α) per element
    grad_Q_b2 = torch.autograd.grad(qg_b2.sum(), Q, retain_graph=True)[0]
    expected_Q_grad = torch.clamp(g.detach(), min=alpha)
    assert torch.allclose(grad_Q_b2, expected_Q_grad, atol=1e-12), (
        "B2 Q-grad must equal max(g, α) elementwise"
    )
    # Floor active: at least one element strictly hit α
    assert (grad_Q_b2 >= alpha - 1e-12).all(), (
        "B2 Q-grad must respect the α floor on every element"
    )
    assert (g.detach() < alpha).any(), (
        "Test setup expected some g_r < α to exercise the floor"
    )
    # τ-grad must match G1 (= Q · g'), unaffected by α
    grad_tau_b2 = torch.autograd.grad(qg_b2.sum(), tau)[0]
    assert torch.allclose(grad_tau_b2, grad_tau_g1, atol=1e-12), (
        "B2 τ-grad must be bit-exact to G1 (= Q·g'), independent of α"
    )
