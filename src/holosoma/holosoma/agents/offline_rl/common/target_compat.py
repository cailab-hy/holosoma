"""Step 8 — ``_target_`` compatibility layer.

This module provides a single source of truth for the four
``_target_`` strings that the offline-RL family can take, plus
helpers to:

* ``resolve_target_class(target)`` — turn any *known* target string
  into the agent class object (without going through Hydra).
* ``recommend_target_for_config(config)`` — given a (possibly partial)
    config object/dict, return the recommended *direct* new-style target
    string.
* ``is_legacy_target`` / ``is_direct_target`` — boolean checks.
* ``KNOWN_TARGETS`` — the four supported strings.

Design notes
------------
* This module does **not** modify ``config_values/algo.py`` or rewrite
  any production ``_target_`` string.  Step 8's contract is to make
  *both* the legacy target and the three new direct targets resolve
  safely.
* The legacy target string remains accepted for old configs/checkpoints,
  but resolves directly to :class:`CQLAgent` without importing the
  removed ``offline_cql`` package. New writes should use the direct targets
  returned by ``recommend_target_for_config``; historical compatibility is
  metadata-only and lives in this module.
* All resolver errors raise :class:`ValueError` with the offending
  string interpolated, so failures from Hydra / migration tooling are
  diagnosable.
"""

from __future__ import annotations

from typing import Any, Mapping

# ── Canonical target strings (single source of truth) ─────────────────

LEGACY_TARGET = (
    "holosoma.agents.offline_cql.offline_cql_agent.OfflineCQLAgent"
)
TARGET_CQL = "holosoma.agents.offline_rl.algorithms.cql.agent.CQLAgent"
TARGET_SMQR = "holosoma.agents.offline_rl.algorithms.smqr.agent.SMQRAgent"
TARGET_SMQR_SG = (
    "holosoma.agents.offline_rl.algorithms.smqr_sg.agent.SMQRSGAgent"
)

#: Tuple of every target string this module is willing to resolve.
KNOWN_TARGETS: tuple[str, ...] = (
    LEGACY_TARGET,
    TARGET_CQL,
    TARGET_SMQR,
    TARGET_SMQR_SG,
)

#: Direct (non-legacy) targets — what Step 8+ recommends emitting
#: when *writing* new configs.
DIRECT_TARGETS: tuple[str, ...] = (TARGET_CQL, TARGET_SMQR, TARGET_SMQR_SG)


# ── Boolean predicates ────────────────────────────────────────────────


def is_known_target(target: str) -> bool:
    """Return ``True`` if *target* is one of the four supported strings."""
    return target in KNOWN_TARGETS


def is_legacy_target(target: str) -> bool:
    """Return ``True`` iff *target* is the legacy ``OfflineCQLAgent`` path."""
    return target == LEGACY_TARGET


def is_direct_target(target: str) -> bool:
    """Return ``True`` iff *target* is one of the three direct new agents."""
    return target in DIRECT_TARGETS


# ── String → class resolver ───────────────────────────────────────────


def resolve_target_class(target: str) -> type:
    """Return the agent class object for *target*.

    Equivalent in behaviour to ``hydra.utils.get_class(target)`` for
    the four known strings, but expressed as a closed table so callers
    can rely on it without importing Hydra.

    Raises
    ------
    ValueError
        If *target* is not in :data:`KNOWN_TARGETS`.
    """
    if not isinstance(target, str):
        raise TypeError(
            f"resolve_target_class expects a str, got {type(target).__name__}"
        )

    # Lazy imports keep resolver use cheap and avoid import-time cycles.
    # Step 14: old target metadata resolves to the canonical class without
    # importing the deleted holosoma.agents.offline_cql package.
    if target in (LEGACY_TARGET, TARGET_CQL):
        from holosoma.agents.offline_rl.algorithms.cql.agent import CQLAgent
        return CQLAgent
    if target == TARGET_SMQR:
        from holosoma.agents.offline_rl.algorithms.smqr.agent import SMQRAgent
        return SMQRAgent
    if target == TARGET_SMQR_SG:
        from holosoma.agents.offline_rl.algorithms.smqr_sg.agent import (
            SMQRSGAgent,
        )
        return SMQRSGAgent

    raise ValueError(
        "Unknown _target_ string for offline-RL family: "
        f"{target!r}. Known targets: {KNOWN_TARGETS}"
    )


# ── Config → recommended direct target ────────────────────────────────


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a Mapping- or attribute-style config."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def recommend_target_for_config(config: Any) -> str:
    """Return the *direct* new-style ``_target_`` for *config*.

    Mapping (shared with legacy ``select_offline_rl_agent_class``):

    * ``critic_penalty_mode == 'vanilla_cql'``      → :data:`TARGET_CQL`
    * ``critic_penalty_mode == 'smqr_cont_self'``
      + ``smqr_lse_mode in {'q_times_g', 'q_times_detached_g'}``
                                                      → :data:`TARGET_SMQR`
    * ``critic_penalty_mode == 'smqr_cont_self'``
      + ``smqr_lse_mode in {'sg_weighted_lse', 'sg_blend'}``
                                                   → :data:`TARGET_SMQR_SG`

    Raises
    ------
    ValueError
        If ``critic_penalty_mode`` is unknown or ``smqr_lse_mode`` is
        not one of the four routed variants.
    """
    critic_penalty_mode = _get(config, "critic_penalty_mode")
    smqr_lse_mode = _get(config, "smqr_lse_mode", "q_times_g")

    if critic_penalty_mode == "vanilla_cql":
        return TARGET_CQL

    if critic_penalty_mode == "smqr_cont_self":
        if smqr_lse_mode in ("q_times_g", "q_times_detached_g"):
            return TARGET_SMQR
        if smqr_lse_mode in ("sg_weighted_lse", "sg_blend"):
            return TARGET_SMQR_SG
        raise ValueError(
            "Cannot recommend offline-RL target: unsupported "
            f"smqr_lse_mode={smqr_lse_mode!r} for "
            "critic_penalty_mode='smqr_cont_self'. "
            "Supported: 'q_times_g', 'q_times_detached_g' (→ SMQRAgent); "
            "'sg_weighted_lse', 'sg_blend' (→ SMQRSGAgent). "
            "Learned-τ / historical variants are not routed."
        )

    raise ValueError(
        "Cannot recommend offline-RL target: unsupported "
        f"critic_penalty_mode={critic_penalty_mode!r}. "
        "Supported: 'vanilla_cql' (→ CQLAgent), "
        "'smqr_cont_self' (→ SMQRAgent / SMQRSGAgent)."
    )


# ── Migration helpers ────────────────────────────────────────────────


def migrate_target_string(old_target: str, config: Any) -> tuple[str, str]:
    """Return ``(new_target, reason)`` for a legacy/known *old_target*.

    * ``new_target`` is the direct new-style target string.
    * ``reason`` is a short human-readable string explaining the
      decision (used by the migration script's manifest).

    Same rules as :func:`recommend_target_for_config` for the legacy
    target.  If *old_target* is already a direct target, it is
    returned unchanged with reason ``'already direct'``.  Unknown
    targets raise :class:`ValueError`.
    """
    if old_target == LEGACY_TARGET:
        new = recommend_target_for_config(config)
        cp = _get(config, "critic_penalty_mode")
        lse = _get(config, "smqr_lse_mode", "q_times_g")
        reason = (
            f"legacy→direct; critic_penalty_mode={cp!r}, "
            f"smqr_lse_mode={lse!r}"
        )
        return new, reason

    if old_target in DIRECT_TARGETS:
        return old_target, "already direct"

    raise ValueError(
        f"Cannot migrate unknown _target_ string: {old_target!r}. "
        f"Known targets: {KNOWN_TARGETS}"
    )
