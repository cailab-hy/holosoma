"""Offline-RL algorithm registry (Step 5 scaffold).

Purpose
-------
Provide a **lookup-only** registry that future baseline implementations
can be added to without touching ``train_agent.py``.  The registry is intentionally NOT connected to
``train_agent.py`` in this phase — wiring is deferred to a later
integration step.

Status flags
------------
* ``"legacy"``      — entry originated in the historical monolithic
                      offline-RL implementation.  Step 9
                      removed the ``legacy_*_module`` metadata fields;
                      resolve old dotted paths via
                      ``target_compat.LEGACY_TARGET`` when needed.
* ``"placeholder"`` — entry has no implementation yet.  Calling
                      :func:`get_algorithm_entry` returns the metadata
                      dict, but :func:`instantiate_algorithm` raises
                      ``NotImplementedError``.

Supported keys (this phase)
---------------------------
=================  ============  ==========================================
key                status        canonical implementation
=================  ============  ==========================================
``cql``            legacy        algorithms.cql.agent.CQLAgent
``smqr``           legacy        algorithms.smqr.agent.SMQRAgent
``smqr_sg``        legacy        algorithms.smqr_sg.agent.SMQRSGAgent
``bc``             placeholder   \u2014
``iql``            placeholder   \u2014
``td3_bc``         placeholder   \u2014
``awac``           placeholder   \u2014
=================  ============  ==========================================

The legacy entries are listed so downstream tooling (e.g. eval
manifest validators) can resolve algorithm names to entry metadata
without instantiating anything.  The placeholders document the future
roadmap from ``docs/offline_rl_refactor_plan.md`` \u00a710.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AlgorithmEntry:
    """Metadata describing a registered algorithm.

    Step-5 fields (unchanged, retained for backward compatibility):
        ``name``, ``status``, ``description``, ``legacy_config_keys``,
        ``helper_modules``, ``notes``.

    Step-9 note:
        The ``legacy_agent_module`` metadata field was removed.
        Callers that need the legacy dotted path should import
        :data:`holosoma.agents.offline_rl.common.target_compat.LEGACY_TARGET`.

    Step-7 additions (opt-in metadata; defaults are no-ops so all
    existing tests continue to pass):
        family
            One of ``"cql"``, ``"smqr"``, ``"smqr_sg"`` for legacy
            entries; ``None`` for placeholders.  Used by the opt-in
            runner to disambiguate ``smqr`` vs ``smqr_sg`` family-level
            knobs.
        required_config_keys
            Subset of ``legacy_config_keys`` that the runner MUST
            forward verbatim onto ``train_agent.py``.  Empty for
            placeholders.  Always a subset of ``legacy_config_keys``.
        default_dataset_key
            Canonical dataset registry key the algorithm pairs with
            by convention.  ``None`` for placeholders.
        recommended_config_example
            Repository-relative path to a YAML example documenting
            the recommended hyper-parameters (S6 best for smqr_sg).
        train_script_reference
            Repository-relative path to the canonical historical launch
            script (replication shell).  Documentation only.
        eval_manifest_reference
            Repository-relative path to the eval manifest entry that
            covers this algorithm.  Documentation only.

    The registry still does **not** wire ``instantiate_algorithm`` to
    return a real agent; entries raise ``NotImplementedError`` to steer
    callers toward the documented train path (preserving the production
    train_agent.py contract intact).
    """

    name: str
    status: str
    description: str
    legacy_config_keys: Mapping[str, Any] = field(default_factory=dict)
    helper_modules: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    # ── Step-7 additions ────────────────────────────────────────────
    family: str | None = None
    required_config_keys: tuple[str, ...] = field(default_factory=tuple)
    default_dataset_key: str | None = None
    recommended_config_example: str | None = None
    train_script_reference: str | None = None
    eval_manifest_reference: str | None = None


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────
_ALLOWED_STATUS = {"legacy", "placeholder"}
_REGISTRY: dict[str, AlgorithmEntry] = {}


def _register(entry: AlgorithmEntry) -> None:
    if entry.status not in _ALLOWED_STATUS:
        raise ValueError(
            f"AlgorithmEntry.status must be one of {_ALLOWED_STATUS}, "
            f"got {entry.status!r}"
        )
    if entry.name in _REGISTRY:
        raise ValueError(f"algorithm {entry.name!r} already registered")
    _REGISTRY[entry.name] = entry


# ── Legacy entries (cql / smqr / smqr_sg) ──────────────────────────
_register(AlgorithmEntry(
    name="cql",
    status="legacy",
    description="Vanilla Conservative Q-Learning.",
    legacy_config_keys={
        "critic_penalty_mode": "vanilla_cql",
    },
    helper_modules=(
        "holosoma.agents.offline_rl.algorithms.cql.losses",
    ),
    notes=(
        "Conservative penalty helper extracted in Step 4-A; canonical "
        "runtime class is CQLAgent. Registry-based construction is still "
        "not wired."
    ),
    family="cql",
    required_config_keys=("critic_penalty_mode",),
    default_dataset_key="wbt_object",
    recommended_config_example="configs/offline_rl/cql_wbt_object.yaml",
    train_script_reference="scripts/train_replication/train_cql_seed2.sh",
    eval_manifest_reference="configs/eval/offline_rl_eval_manifest.yaml",
))

_register(AlgorithmEntry(
    name="smqr",
    status="legacy",
    description=(
        "Soft-Max Q-Regularization, anchor-only vanilla q_times_g "
        "variant (no SG)."
    ),
    legacy_config_keys={
        "critic_penalty_mode": "smqr_cont_self",
        "algo_mode": "smqr_anchor",
        "smqr_anchor_objective": "vanilla",
        "sc_tau_res_scale": 0.0,
        "smqr_lse_mode": "q_times_g",
    },
    helper_modules=(
        "holosoma.agents.offline_rl.algorithms.smqr.losses",
    ),
    notes=(
        "q_times_g penalty extracted in Step 4-B.  q_times_detached_g "
        "ablation plus learned-tau/stabilised variants are OUT OF SCOPE "
        "for the current canonical SMQRAgent route."
    ),
    family="smqr",
    required_config_keys=(
        "critic_penalty_mode",
        "algo_mode",
        "smqr_anchor_objective",
        "sc_tau_res_scale",
        "smqr_lse_mode",
    ),
    default_dataset_key="wbt_object",
    recommended_config_example="configs/offline_rl/smqr_wbt_object.yaml",
    train_script_reference="scripts/train_replication/train_smqr_seed2.sh",
    eval_manifest_reference="configs/eval/offline_rl_eval_manifest.yaml",
))

_register(AlgorithmEntry(
    name="smqr_sg",
    status="legacy",
    description=(
        "SMQR with Stop-Gradient gate: sg_weighted_lse + sg_blend "
        "(LOSS-level convex blend, schedule-driven \u03bb(t))."
    ),
    legacy_config_keys={
        "critic_penalty_mode": "smqr_cont_self",
        "algo_mode": "smqr_anchor",
        "smqr_anchor_objective": "vanilla",
        "sc_tau_res_scale": 0.0,
        "smqr_lse_mode": "sg_blend",   # or 'sg_weighted_lse'
    },
    helper_modules=(
        "holosoma.agents.offline_rl.algorithms.smqr_sg.losses",
        "holosoma.agents.offline_rl.algorithms.smqr_sg.schedules",
    ),
    notes=(
        "Current best golden: exp_S6_1m_fixed05_cqls05_bc30_seed1_bs4096 "
        "(sg_blend, fixed \u03bb=0.5, cql_loss_scale=0.5, bc_weight=3.0).  "
        "Loss & schedule helpers extracted in Step 4-C."
    ),
    family="smqr_sg",
    required_config_keys=(
        "critic_penalty_mode",
        "algo_mode",
        "smqr_anchor_objective",
        "sc_tau_res_scale",
        "smqr_lse_mode",
    ),
    default_dataset_key="wbt_object",
    recommended_config_example="configs/offline_rl/smqr_sg_wbt_object.yaml",
    train_script_reference="scripts/train_replication/_common.sh",
    eval_manifest_reference="configs/eval/offline_rl_eval_manifest.yaml",
))

# ── Placeholder entries (no implementation yet) ────────────────────
_register(AlgorithmEntry(
    name="bc",
    status="placeholder",
    description="Behaviour Cloning baseline (not yet implemented).",
    notes="Planned baseline; see docs/offline_rl_refactor_plan.md \u00a710.",
))
_register(AlgorithmEntry(
    name="iql",
    status="placeholder",
    description="Implicit Q-Learning (Kostrikov et al. 2021) — not implemented.",
    notes="Planned baseline; see docs/offline_rl_refactor_plan.md \u00a710.",
))
_register(AlgorithmEntry(
    name="td3_bc",
    status="placeholder",
    description="TD3+BC (Fujimoto & Gu 2021) — not implemented.",
    notes="Planned baseline; see docs/offline_rl_refactor_plan.md \u00a710.",
))
_register(AlgorithmEntry(
    name="awac",
    status="placeholder",
    description="Advantage-Weighted Actor-Critic (Nair et al. 2020) — not implemented.",
    notes="Planned baseline; see docs/offline_rl_refactor_plan.md \u00a710.",
))


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def get_algorithm_entry(name: str) -> AlgorithmEntry:
    """Look up algorithm metadata by canonical key.

    Raises
    ------
    KeyError
        If the name is not registered at all.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_algorithms() -> list[str]:
    """Return the sorted list of registered algorithm names."""
    return sorted(_REGISTRY)


def instantiate_algorithm(name: str, *args: Any, **kwargs: Any) -> Any:
    """Construct an algorithm instance (placeholder).

    This is intentionally NOT wired in Step 5.  ``legacy`` entries
    raise ``NotImplementedError`` because production construction is
    still served by ``train_agent.py`` directly; ``placeholder``
    entries raise ``NotImplementedError`` because the algorithm has
    no implementation yet.  Future integration phases will replace
    this with a real factory.
    """
    entry = get_algorithm_entry(name)
    if entry.status == "legacy":
        raise NotImplementedError(
            f"algorithm {name!r} is implemented by canonical offline_rl "
            "agents; registry-based "
            "instantiation is not wired in this scaffold phase. "
            "Use train_agent.py with the documented config flags."
        )
    raise NotImplementedError(
        f"algorithm {name!r} has status={entry.status!r} — "
        f"no implementation available."
    )


__all__ = [
    "AlgorithmEntry",
    "get_algorithm_entry",
    "instantiate_algorithm",
    "list_algorithms",
]
