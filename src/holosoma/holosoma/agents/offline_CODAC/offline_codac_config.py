from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CODACConfig:
    """CODAC-specific hyperparameters.

    These are separate from ``FastSACConfig`` to avoid polluting the base
    config.  ``FastSACConfig.conservative_weight`` is still used as the
    global multiplier for the conservative critic loss term.

    Naming follows ``FastSACConfig`` conventions (snake_case, docstring
    on the line after each field).
    """

    # ------------------------------------------------------------------
    # Conservative penalty — action sampling
    # ------------------------------------------------------------------

    conservative_coef: float = 5.0
    """Coefficient for the conservative critic penalty term.

    This is the *CODAC-side* weight.  It is multiplied **in addition to**
    ``FastSACConfig.conservative_weight`` so that the total conservative
    loss seen by the critic is::

        conservative_weight * conservative_coef * raw_penalty

    Set ``FastSACConfig.conservative_weight = 1.0`` and tune this field,
    **or** set this to 1.0 and tune ``conservative_weight`` — either works.
    The two-level design keeps the parent hook contract intact.
    """

    num_conservative_actions: int = 10
    """Number of OOD actions to sample **per source** (random, current-policy,
    next-policy) for the conservative penalty.  Total OOD actions per obs =
    ``num_conservative_actions * (number of enabled sources)``."""

    conservative_action_sample_mode: str = "random_policy"
    """Which action sources to include in the OOD set.

    Supported values:
    - ``"random"`` — uniform-random actions only.
    - ``"policy"`` — current-policy actions only.
    - ``"random_policy"`` — both random and current-policy (CQL / CODAC default).
    - ``"random_policy_next"`` — random + current-policy + next-obs policy.
    """

    # ------------------------------------------------------------------
    # Conservative penalty — distributional / risk
    # ------------------------------------------------------------------

    codac_risk_mode: str = "neutral"
    """How to aggregate the distributional Q to a scalar for the penalty.

    - ``"neutral"`` — expectation E[Z] (standard CQL-style).
    - ``"cvar"`` — CVaR_α  (risk-sensitive, pessimistic).
    - ``"wang"`` — Wang risk measure.
    - ``"power"`` — CPW power distortion.
    - ``"quantile"`` — specific quantile τ.

    Only ``"neutral"`` is implemented in the first pass; the rest are
    extension points for risk-sensitive CODAC variants.
    """

    codac_risk_param: float = 1.0
    """Parameter for the risk measure selected by ``codac_risk_mode``.

    Interpretation depends on the mode:
    - ``"neutral"`` — ignored.
    - ``"cvar"`` — α ∈ (0, 1]; smaller = more pessimistic.
    - ``"wang"`` — η; negative = risk-averse.
    - ``"power"`` — exponent.
    - ``"quantile"`` — τ ∈ (0, 1].
    """

    # ------------------------------------------------------------------
    # Conservative penalty — target computation
    # ------------------------------------------------------------------

    codac_target_mode: str = "mean"
    """How to combine Q-values across the Q-ensemble for the penalty.

    - ``"mean"``  — average over ``num_q_networks`` (default).
    - ``"min"``   — pessimistic (min over ensemble).
    - ``"individual"`` — compute penalty per Q-network, then sum.
    """

    conservative_temp: float = 1.0
    """Temperature (τ) for the logsumexp over OOD Q-values::

        logsumexp(Q_ood / τ) × τ

    Larger τ → softer max (closer to mean); τ → 0 → hard max.
    """

    # ------------------------------------------------------------------
    # Actor regularization
    # ------------------------------------------------------------------

    actor_bc_coef: float = 0.0
    """Behavior-cloning regularization coefficient for the actor.

    When > 0, adds ``actor_bc_coef * MSE(policy_action, dataset_action)``
    to the actor loss.  0 disables it (pure SAC actor under CODAC critic).
    """

    # ------------------------------------------------------------------
    # Lagrange (auto-tune conservative_coef)
    # ------------------------------------------------------------------

    use_lagrange: bool = False
    """Whether to auto-tune ``conservative_coef`` via a Lagrange multiplier.

    When False, ``conservative_coef`` is used as a fixed weight.
    (TODO: implement in a later stage.)
    """

    lagrange_threshold: float = 10.0
    """Target threshold τ_α for the Lagrange constraint.

    The Lagrange dual updates ``conservative_coef`` so that the
    conservative gap ≈ ``lagrange_threshold``.
    Only active when ``use_lagrange = True``.
    """

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    log_codac_debug_metrics: bool = True
    """Whether to log CODAC-specific diagnostic metrics to TensorBoard.

    When True, the following extra scalars are logged:
    - ``codac/conservative_penalty``  (raw penalty before coef)
    - ``codac/q_dataset_mean``
    - ``codac/q_ood_random_mean``
    - ``codac/q_ood_policy_mean``
    - ``codac/conservative_gap``  (ood_mean − dataset_mean)
    """
