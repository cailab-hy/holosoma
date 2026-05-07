"""Unified algorithm-mode resolver for Offline CQL.

Phase A scaffold (no behavioural change).
========================================

This module introduces a thin *router* over the existing legacy keys
(``critic_penalty_mode``, ``sc_tau_res_scale``, ...) so that the three
research tracks can be addressed by a single, explicit name:

* ``cql``           — vanilla CQL (legacy ``critic_penalty_mode='vanilla_cql'``)
* ``smqr_anchor``   — anchor-only SMQR (legacy ``smqr_cont_self`` + ``sc_tau_res_scale==0.0``)
* ``smqr_learned``  — learnable τ-residual SMQR (legacy ``smqr_cont_self`` + ``sc_tau_res_scale>0``)

Backward compatibility
----------------------
* ``algo_mode='auto'`` (default) preserves the legacy semantics
  bit-for-bit: nothing is rewritten, the resolver only *labels* the run.
* Explicit ``algo_mode=<name>`` is validated against the legacy keys; on
  conflict the resolver fails fast.

Phase A guard
-------------
``smqr_learned`` is recognised but **must not run** in Phase A.
``assert_phase_a_compatible`` raises ``NotImplementedError`` so that
optimizer-level execution of learned-τ training is blocked until the
Phase B branch lands explicitly.

This file intentionally has *no* dependencies on torch / pydantic so it
can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ── Mode names ────────────────────────────────────────────────────────
MODE_AUTO = "auto"
MODE_CQL = "cql"
MODE_SMQR_ANCHOR = "smqr_anchor"
MODE_SMQR_LEARNED = "smqr_learned"

ALLOWED_MODES = (MODE_AUTO, MODE_CQL, MODE_SMQR_ANCHOR, MODE_SMQR_LEARNED)
ALLOWED_LEARNED_VARIANTS = (
    "vanilla",
    "stabilized",
    "v1_oneside_shrink",
    "f1_st_qg",  # Phase F: V1 τ-param + ST-split Q·g objective (no log(g+ε)).
)

# Phase E (objective-isolation ablation): orthogonal selector that
# governs the SMQR critic-loss objective form when ``algo_mode='smqr_anchor'``.
# ``vanilla``    — the existing  Q · g − log p   weighted-logits form (default; bit-equivalent).
# ``stabilized`` — the Phase C   Q + log(g+ε) − log p   stabilised form, but with τ ≡ τ_anchor
#                  (no learned residual / head — `sc_tau_res_scale=0.0`).
# Used to isolate whether the stabilised objective itself is the 1st-order
# cause of the Phase C/D failures, independently of τ-parameterisation.
ALLOWED_ANCHOR_OBJECTIVES = ("vanilla", "stabilized")

# ── τ source names ────────────────────────────────────────────────────
TAU_NONE = "none"
TAU_ANCHOR = "anchor"
TAU_LEARNED = "learned"

# ── Legacy critic_penalty_mode values that this resolver knows about ──
LEGACY_VANILLA_CQL = "vanilla_cql"
LEGACY_SMQR = "smqr_cont_self"

# ── Run-name short tags (for filesystem safety) ──────────────────────
_MODE_TAG = {
    MODE_CQL: "cql",
    MODE_SMQR_ANCHOR: "smqranc",
    MODE_SMQR_LEARNED: "smqrlrn",
}


@dataclass(frozen=True)
class ResolvedAlgoMode:
    """Resolution result returned by :func:`resolve_algo_mode`.

    Attributes
    ----------
    mode:
        One of ``cql`` / ``smqr_anchor`` / ``smqr_learned`` (never
        ``auto`` — that is resolved away).
    tau_source:
        ``none`` / ``anchor`` / ``learned``.  Diagnostic label only;
        the actual τ computation still lives in the legacy code path
        in Phase A.
    legacy_critic_penalty_mode:
        The string consumed by the existing critic-loss branch
        (``vanilla_cql`` or ``smqr_cont_self``).  Phase A does NOT
        rewrite ``args.critic_penalty_mode``; this field is provided
        as a sanity check / future hand-off point.
    tau_res_scale:
        Mirror of ``args.sc_tau_res_scale`` (only meaningful for SMQR).
    learned_variant:
        ``vanilla`` (default) or ``stabilized`` (placeholder; raises
        if invoked under Phase A guard with learned mode).
    logging_prefix:
        TensorBoard / metric prefix, e.g. ``train/smqr_anchor/``.
    run_name_tag:
        Short, filesystem-safe tag, e.g. ``smqranc`` (used only by
        :func:`format_run_name`).
    explicit:
        ``True`` iff the user passed ``algo_mode`` explicitly (i.e.
        not via legacy inference).
    """

    mode: str
    tau_source: str
    legacy_critic_penalty_mode: str
    tau_res_scale: float
    learned_variant: str
    logging_prefix: str
    run_name_tag: str
    explicit: bool


def _get(args: Any, name: str, default: Any) -> Any:
    """Tolerant getattr for both dataclasses and SimpleNamespace-like objs."""
    return getattr(args, name, default)


def _infer_from_legacy(
    legacy_penalty_mode: str, tau_res_scale: float
) -> str:
    """Infer the new mode label from the legacy keys."""
    if legacy_penalty_mode == LEGACY_VANILLA_CQL:
        return MODE_CQL
    if legacy_penalty_mode == LEGACY_SMQR:
        # scale==0 is bit-equivalent to anchor-only SMQR (the current
        # hypothesis track). scale>0 means τ-head residual is active →
        # learned-τ branch.
        if float(tau_res_scale) == 0.0:
            return MODE_SMQR_ANCHOR
        return MODE_SMQR_LEARNED
    # Any other legacy mode (e.g. sc_cql) is out-of-scope for this
    # router; we surface it as a CQL-family run for logging purposes
    # but flag it as "explicit-required" by raising on unknown modes
    # only when the user *explicitly* requests one of our three.  When
    # auto-resolving, we conservatively map unknowns to MODE_CQL so the
    # legacy run path is unchanged — the resolver does NOT alter
    # behaviour, only labels.
    return MODE_CQL


def _validate_explicit_against_legacy(
    explicit_mode: str, legacy_penalty_mode: str, tau_res_scale: float
) -> None:
    """Fail fast when an explicit ``algo_mode`` contradicts legacy keys.

    Only validates the cases this router actively claims ownership of.
    """
    if explicit_mode == MODE_CQL:
        if legacy_penalty_mode not in (LEGACY_VANILLA_CQL,):
            raise ValueError(
                f"algo_mode='cql' but critic_penalty_mode="
                f"'{legacy_penalty_mode}'. Set critic_penalty_mode="
                f"'{LEGACY_VANILLA_CQL}' or leave algo_mode='auto'."
            )
    elif explicit_mode == MODE_SMQR_ANCHOR:
        if legacy_penalty_mode != LEGACY_SMQR:
            raise ValueError(
                f"algo_mode='smqr_anchor' requires critic_penalty_mode="
                f"'{LEGACY_SMQR}', got '{legacy_penalty_mode}'."
            )
        if float(tau_res_scale) != 0.0:
            raise ValueError(
                f"algo_mode='smqr_anchor' requires sc_tau_res_scale=0.0 "
                f"(anchor-only). Got sc_tau_res_scale={tau_res_scale}. "
                f"Use algo_mode='smqr_learned' for non-zero residual."
            )
    elif explicit_mode == MODE_SMQR_LEARNED:
        if legacy_penalty_mode != LEGACY_SMQR:
            raise ValueError(
                f"algo_mode='smqr_learned' requires critic_penalty_mode="
                f"'{LEGACY_SMQR}', got '{legacy_penalty_mode}'."
            )
        if float(tau_res_scale) <= 0.0:
            raise ValueError(
                f"algo_mode='smqr_learned' requires sc_tau_res_scale>0. "
                f"Got sc_tau_res_scale={tau_res_scale}. "
                f"Use algo_mode='smqr_anchor' for the anchor-only track."
            )


def resolve_algo_mode(args: Any) -> ResolvedAlgoMode:
    """Resolve the unified ``algo_mode`` for Phase A.

    Reads three new keys (``algo_mode``, ``smqr_learned_variant``,
    ``smqr_logging_namespace``) plus the legacy keys
    (``critic_penalty_mode``, ``sc_tau_res_scale``) and returns a
    :class:`ResolvedAlgoMode`.

    Notes
    -----
    * In ``algo_mode='auto'`` the legacy keys remain authoritative;
      this preserves bit-equivalence with all pre-existing runs.
    * No mutation of ``args`` is performed.
    * No torch / module construction occurs here — pure routing.
    """
    raw_mode = str(_get(args, "algo_mode", MODE_AUTO)).strip().lower()
    if raw_mode not in ALLOWED_MODES:
        raise ValueError(
            f"algo_mode='{raw_mode}' is not one of {ALLOWED_MODES}."
        )

    legacy_penalty_mode = str(_get(args, "critic_penalty_mode", LEGACY_VANILLA_CQL))
    tau_res_scale = float(_get(args, "sc_tau_res_scale", 0.0))

    explicit = raw_mode != MODE_AUTO
    if explicit:
        _validate_explicit_against_legacy(raw_mode, legacy_penalty_mode, tau_res_scale)
        mode = raw_mode
    else:
        mode = _infer_from_legacy(legacy_penalty_mode, tau_res_scale)

    # Variant placeholder — Phase A only allows "vanilla" to be invoked
    # downstream (and even that only for cql/smqr_anchor; smqr_learned
    # is blocked entirely by `assert_phase_a_compatible`).
    raw_variant = str(_get(args, "smqr_learned_variant", "vanilla")).strip().lower()
    if raw_variant not in ALLOWED_LEARNED_VARIANTS:
        raise ValueError(
            f"smqr_learned_variant='{raw_variant}' is not one of "
            f"{ALLOWED_LEARNED_VARIANTS}."
        )

    # τ source label
    if mode == MODE_CQL:
        tau_source = TAU_NONE
    elif mode == MODE_SMQR_ANCHOR:
        tau_source = TAU_ANCHOR
    else:  # MODE_SMQR_LEARNED
        tau_source = TAU_LEARNED

    # Logging prefix — user override wins; otherwise derive from mode.
    user_ns: Optional[str] = _get(args, "smqr_logging_namespace", None)
    if user_ns:
        logging_prefix = str(user_ns).rstrip("/") + "/"
    else:
        logging_prefix = f"train/{mode}/"

    run_name_tag = _MODE_TAG[mode]

    return ResolvedAlgoMode(
        mode=mode,
        tau_source=tau_source,
        legacy_critic_penalty_mode=legacy_penalty_mode,
        tau_res_scale=tau_res_scale,
        learned_variant=raw_variant,
        logging_prefix=logging_prefix,
        run_name_tag=run_name_tag,
        explicit=explicit,
    )


def assert_phase_a_compatible(
    resolved: ResolvedAlgoMode,
    *,
    allow_learned: bool = False,
    allow_stabilized: bool = False,
    allow_v1: bool = False,
    allow_f1: bool = False,
    anchor_objective: str = "vanilla",
    allow_anchor_stab: bool = False,
) -> None:
    """Phase-A/B/C guard: gate any execution path that would *train* learned-τ.

    Phase A (default, ``allow_learned=False``)
        Any ``smqr_learned`` resolution raises ``NotImplementedError``.

    Phase B (``allow_learned=True``, ``allow_stabilized=False``)
        ``smqr_learned`` with ``learned_variant='vanilla'`` is permitted.
        ``learned_variant='stabilized'`` remains blocked.

    Phase C (``allow_learned=True`` AND ``allow_stabilized=True``)
        ``smqr_learned`` with ``learned_variant='stabilized'`` is
        permitted.  The stabilised path replaces the vanilla
        ``logsumexp(Q·g - log p)`` factor with the softmax-bounded
        ``logsumexp(Q + log(g+ε) - log p)`` formulation; it is
        otherwise structurally identical to the vanilla learned-τ
        branch (same τ-head, same anchor, same scale).

        Stabilized also requires ``allow_learned=True`` (it is a
        sub-mode of ``smqr_learned``).  Stabilized is invalid for
        any non-learned mode (cql / smqr_anchor).

    Parameters
    ----------
    resolved:
        Output of :func:`resolve_algo_mode`.
    allow_learned:
        Phase B opt-in.  Wire to ``smqr_learned_phase_b_optin``.
    allow_stabilized:
        Phase C opt-in.  Wire to ``smqr_learned_phase_c_optin``.
        Has no effect unless ``allow_learned=True`` AND
        ``learned_variant='stabilized'``.
    allow_v1:
        Phase D opt-in.  Wire to ``smqr_learned_phase_d_optin``.
    anchor_objective:
        Phase E selector for the anchor-only SMQR objective form.
        ``'vanilla'`` (default) is bit-equivalent to the legacy
        anchor-only path.  ``'stabilized'`` switches the
        weighted-logits to ``Q + log(g+ε) − log p`` while keeping
        ``τ ≡ τ_anchor`` (no learned residual).  Only valid with
        ``algo_mode='smqr_anchor'``.
    allow_anchor_stab:
        Phase E opt-in.  Wire to ``smqr_anchor_phase_e_optin``.
        Has no effect unless ``anchor_objective='stabilized'``.

    Raises
    ------
    NotImplementedError
        If the resolution would invoke a still-gated branch.
    """
    if resolved.mode == MODE_SMQR_LEARNED and not allow_learned:
        raise NotImplementedError(
            "smqr_learned training is gated. "
            "Resolved mode='smqr_learned' (legacy critic_penalty_mode="
            f"'{resolved.legacy_critic_penalty_mode}', "
            f"sc_tau_res_scale={resolved.tau_res_scale}). "
            "Phase B opt-in required: pass "
            "--algo.config.smqr-learned-phase-b-optin true "
            "(see scripts/train_replication/smoke_smqr_learned.sh). "
            "To run the anchor-only hypothesis track, use "
            "algo_mode='smqr_anchor' with sc_tau_res_scale=0.0."
        )
    if resolved.learned_variant == "stabilized":
        # Stabilized is a sub-mode of smqr_learned only.
        if resolved.mode != MODE_SMQR_LEARNED:
            raise NotImplementedError(
                "smqr_learned_variant='stabilized' is only valid with "
                "algo_mode='smqr_learned'.  Got mode="
                f"'{resolved.mode}'."
            )
        if not allow_stabilized:
            raise NotImplementedError(
                "smqr_learned_variant='stabilized' is gated. "
                "Phase C opt-in required: pass "
                "--algo.config.smqr-learned-phase-c-optin true "
                "(in addition to --algo.config.smqr-learned-phase-b-optin true). "
                "See scripts/train_replication/smoke_smqr_learned_stab.sh."
            )
        if not allow_learned:
            # Defence in depth: stabilized cannot run without the
            # Phase B gate also being open (the underlying mode is
            # still smqr_learned).
            raise NotImplementedError(
                "smqr_learned_variant='stabilized' requires both "
                "Phase B and Phase C opt-ins to be set. "
                "Pass --algo.config.smqr-learned-phase-b-optin true "
                "AND --algo.config.smqr-learned-phase-c-optin true."
            )
    if resolved.learned_variant == "v1_oneside_shrink":
        # V1 (Phase D): one-sided residual + small anchor-shrinkage
        # on top of the stabilized objective.  Sub-mode of
        # smqr_learned only.
        if resolved.mode != MODE_SMQR_LEARNED:
            raise NotImplementedError(
                "smqr_learned_variant='v1_oneside_shrink' is only "
                "valid with algo_mode='smqr_learned'.  Got mode="
                f"'{resolved.mode}'."
            )
        if not allow_v1:
            raise NotImplementedError(
                "smqr_learned_variant='v1_oneside_shrink' is gated. "
                "Phase D opt-in required: pass "
                "--algo.config.smqr-learned-phase-d-optin true "
                "(in addition to --algo.config.smqr-learned-phase-b-optin true). "
                "See scripts/train_replication/smoke_smqr_learned_v1.sh."
            )
        if not allow_learned:
            raise NotImplementedError(
                "smqr_learned_variant='v1_oneside_shrink' requires "
                "both Phase B and Phase D opt-ins to be set. "
                "Pass --algo.config.smqr-learned-phase-b-optin true "
                "AND --algo.config.smqr-learned-phase-d-optin true."
            )
    if resolved.learned_variant == "f1_st_qg":
        # Phase F: V1 τ-parameterisation (one-sided softplus + shrinkage)
        # combined with the ST-split Q·g objective.  Sub-mode of
        # smqr_learned only.  Forward value is bit-exact equal to
        # vanilla ``Q·g``; backward halves the Q·g' contribution to
        # both θ_Q and θ_τ via a symmetric stop-gradient identity.
        if resolved.mode != MODE_SMQR_LEARNED:
            raise NotImplementedError(
                "smqr_learned_variant='f1_st_qg' is only valid with "
                f"algo_mode='smqr_learned'.  Got mode='{resolved.mode}'."
            )
        if not allow_f1:
            raise NotImplementedError(
                "smqr_learned_variant='f1_st_qg' is gated. "
                "Phase F opt-in required: pass "
                "--algo.config.smqr-learned-phase-f-optin true "
                "(in addition to --algo.config.smqr-learned-phase-b-optin true). "
                "See scripts/train_replication/smoke_smqr_learned_f1.sh."
            )
        if not allow_learned:
            raise NotImplementedError(
                "smqr_learned_variant='f1_st_qg' requires both "
                "Phase B and Phase F opt-ins to be set. "
                "Pass --algo.config.smqr-learned-phase-b-optin true "
                "AND --algo.config.smqr-learned-phase-f-optin true."
            )

    # ── Phase E: anchor-only + stabilised objective (objective-isolation) ──
    # Orthogonal to the learned-τ variants above.  ``anchor_objective``
    # is consulted only when the resolved mode is ``smqr_anchor`` —
    # for any other mode, ``stabilized`` is invalid.
    ao = str(anchor_objective).strip().lower()
    if ao not in ALLOWED_ANCHOR_OBJECTIVES:
        raise ValueError(
            f"smqr_anchor_objective='{ao}' is not one of "
            f"{ALLOWED_ANCHOR_OBJECTIVES}."
        )
    if ao == "stabilized":
        if resolved.mode != MODE_SMQR_ANCHOR:
            raise NotImplementedError(
                "smqr_anchor_objective='stabilized' is only valid with "
                "algo_mode='smqr_anchor' (objective-isolation ablation). "
                f"Got mode='{resolved.mode}'."
            )
        if not allow_anchor_stab:
            raise NotImplementedError(
                "smqr_anchor_objective='stabilized' is gated. "
                "Phase E opt-in required: pass "
                "--algo.config.smqr-anchor-phase-e-optin true. "
                "See scripts/train_replication/smoke_smqr_anchor_stab.sh."
            )


def format_run_name(base: str, resolved: ResolvedAlgoMode, seed: int) -> str:
    """Compose a mode-prefixed, seed-suffixed run name.

    Convention (Phase A scaffold)::

        <base>__mode-<tag>__seed<S>

    The double-underscore separator makes the mode/seed segments easy to
    parse for downstream report scripts and avoids collisions with the
    pre-existing ``exp_NN_<algo>_<tag>_seed<S>`` naming used by the
    hypothesis track scripts (which are NOT modified in Phase A).
    """
    return f"{base}__mode-{resolved.run_name_tag}__seed{int(seed)}"
