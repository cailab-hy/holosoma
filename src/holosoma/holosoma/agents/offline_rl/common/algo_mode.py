"""Unified algorithm-mode resolver for canonical offline-RL.

This module owns the lightweight, torch-free resolver that labels legacy
offline-RL config combinations as one of the canonical algorithm modes:

* ``cql``           — vanilla CQL (``critic_penalty_mode='vanilla_cql'``)
* ``smqr_anchor``   — anchor-only SMQR (``smqr_cont_self`` + ``sc_tau_res_scale==0.0``)
* ``smqr_learned``  — learned-τ SMQR variants (currently gated)

The removed ``holosoma.agents.offline_cql`` package used to host this helper.
It now lives here so tests and future tooling can keep using the resolver
without restoring the legacy package.
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
    "f1_st_qg",
)

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
    """Resolution result returned by :func:`resolve_algo_mode`."""

    mode: str
    tau_source: str
    legacy_critic_penalty_mode: str
    tau_res_scale: float
    learned_variant: str
    logging_prefix: str
    run_name_tag: str
    explicit: bool


def _get(args: Any, name: str, default: Any) -> Any:
    """Tolerant ``getattr`` for dataclasses and ``SimpleNamespace`` objects."""
    return getattr(args, name, default)


def _infer_from_legacy(
    legacy_penalty_mode: str, tau_res_scale: float
) -> str:
    """Infer the canonical mode label from legacy config keys."""
    if legacy_penalty_mode == LEGACY_VANILLA_CQL:
        return MODE_CQL
    if legacy_penalty_mode == LEGACY_SMQR:
        if float(tau_res_scale) == 0.0:
            return MODE_SMQR_ANCHOR
        return MODE_SMQR_LEARNED
    return MODE_CQL


def _validate_explicit_against_legacy(
    explicit_mode: str, legacy_penalty_mode: str, tau_res_scale: float
) -> None:
    """Fail fast when explicit ``algo_mode`` contradicts legacy keys."""
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
    """Resolve the unified ``algo_mode`` label for an offline-RL config."""
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

    raw_variant = str(_get(args, "smqr_learned_variant", "vanilla")).strip().lower()
    if raw_variant not in ALLOWED_LEARNED_VARIANTS:
        raise ValueError(
            f"smqr_learned_variant='{raw_variant}' is not one of "
            f"{ALLOWED_LEARNED_VARIANTS}."
        )

    if mode == MODE_CQL:
        tau_source = TAU_NONE
    elif mode == MODE_SMQR_ANCHOR:
        tau_source = TAU_ANCHOR
    else:
        tau_source = TAU_LEARNED

    user_ns: Optional[str] = _get(args, "smqr_logging_namespace", None)
    if user_ns:
        logging_prefix = str(user_ns).rstrip("/") + "/"
    else:
        logging_prefix = f"train/{mode}/"

    return ResolvedAlgoMode(
        mode=mode,
        tau_source=tau_source,
        legacy_critic_penalty_mode=legacy_penalty_mode,
        tau_res_scale=tau_res_scale,
        learned_variant=raw_variant,
        logging_prefix=logging_prefix,
        run_name_tag=_MODE_TAG[mode],
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
    """Gate learned/stabilized SMQR variants behind explicit opt-ins."""
    if resolved.mode == MODE_SMQR_LEARNED and not allow_learned:
        raise NotImplementedError(
            "smqr_learned training is gated. "
            "Resolved mode='smqr_learned' (legacy critic_penalty_mode="
            f"'{resolved.legacy_critic_penalty_mode}', "
            f"sc_tau_res_scale={resolved.tau_res_scale}). "
            "Phase B opt-in required: pass "
            "--algo.config.smqr-learned-phase-b-optin true. "
            "To run the anchor-only hypothesis track, use "
            "algo_mode='smqr_anchor' with sc_tau_res_scale=0.0."
        )
    if resolved.learned_variant == "stabilized":
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
                "(in addition to --algo.config.smqr-learned-phase-b-optin true)."
            )
        if not allow_learned:
            raise NotImplementedError(
                "smqr_learned_variant='stabilized' requires both "
                "Phase B and Phase C opt-ins to be set."
            )
    if resolved.learned_variant == "v1_oneside_shrink":
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
                "(in addition to --algo.config.smqr-learned-phase-b-optin true)."
            )
        if not allow_learned:
            raise NotImplementedError(
                "smqr_learned_variant='v1_oneside_shrink' requires "
                "both Phase B and Phase D opt-ins to be set."
            )
    if resolved.learned_variant == "f1_st_qg":
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
                "(in addition to --algo.config.smqr-learned-phase-b-optin true)."
            )
        if not allow_learned:
            raise NotImplementedError(
                "smqr_learned_variant='f1_st_qg' requires both "
                "Phase B and Phase F opt-ins to be set."
            )

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
                "--algo.config.smqr-anchor-phase-e-optin true."
            )


def format_run_name(base: str, resolved: ResolvedAlgoMode, seed: int) -> str:
    """Compose a mode-prefixed, seed-suffixed run name."""
    return f"{base}__mode-{resolved.run_name_tag}__seed{int(seed)}"


__all__ = [
    "ALLOWED_ANCHOR_OBJECTIVES",
    "ALLOWED_LEARNED_VARIANTS",
    "ALLOWED_MODES",
    "LEGACY_SMQR",
    "LEGACY_VANILLA_CQL",
    "MODE_AUTO",
    "MODE_CQL",
    "MODE_SMQR_ANCHOR",
    "MODE_SMQR_LEARNED",
    "ResolvedAlgoMode",
    "TAU_ANCHOR",
    "TAU_LEARNED",
    "TAU_NONE",
    "assert_phase_a_compatible",
    "format_run_name",
    "resolve_algo_mode",
]