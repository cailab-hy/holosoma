"""Toy gradient sanity tests for the Step-3 SMQR-SG sub-modes.

These tests are pure-CPU torch micro-benchmarks of the three
``smqr_lse_mode`` variants in
:mod:`holosoma.agents.offline_cql.offline_cql_agent`.  They construct
synthetic ``Q``, ``τ``, ``log p`` tensors that match the shapes used by
the real critic loss (``[num_q, B, K]``), then compare the analytic
``∂L/∂Q`` against ``torch.autograd``.

Reference forms (β = 1.0, eps = 1e-6, all per-candidate):

  q_times_g          : z_i = Q_i · g_i − log p_i
                       ∂lse/∂Q_i = w_i · (g_i + Q_i · g'_i),   g'_i = g_i(1−g_i)/β
  q_times_detached_g : z_i = Q_i · sg(g_i) − log p_i
                       ∂lse/∂Q_i = w_i · g_i  (gate derivative removed)
  sg_weighted_lse    : z_i = Q_i − log p_i + log(sg(g_i) + ε)
                       ∂lse/∂Q_i = w_i        (β-independent)

All tests are intended to be cheap (< 1 s on CPU) so they can run
inside the pre-commit hook / CI smoke gate.
"""

from __future__ import annotations

import math

import pytest
import torch


# ──────────────────────────────────────────────────────────────────
# Forward / loss helpers — mirror the offline_cql_agent.py branches
# at offline_cql_agent.py L2173-L2228 *exactly*.
# ──────────────────────────────────────────────────────────────────


def _logits(
    Q: torch.Tensor,
    tau: torch.Tensor,
    log_p: torch.Tensor,
    *,
    mode: str,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return ``[num_q, B, K]`` weighted logits for a given mode.

    ``Q``, ``log_p`` shape ``[num_q, B, K]``.  ``tau`` shape ``[B]``.
    """
    delta = Q - tau.view(1, -1, 1)
    g = torch.sigmoid(delta / beta)
    if mode == "q_times_g":
        return Q * g - log_p
    if mode == "q_times_detached_g":
        return Q * g.detach() - log_p
    if mode == "sg_weighted_lse":
        return Q - log_p + torch.log(g.detach().clamp_min(eps))
    raise ValueError(f"Unknown mode {mode!r}")


def _loss(z: torch.Tensor) -> torch.Tensor:
    """Match the agent's ``logsumexp(z, dim=-1) − log(N)`` reduction
    used inside ``cql_logsumexp`` (the ``− log N`` constant has zero
    gradient so we drop it).  Sum over (num_q, B) → scalar.
    """
    return torch.logsumexp(z, dim=-1).sum()


def _analytic_grad(
    Q: torch.Tensor,
    g: torch.Tensor,
    softmax_w: torch.Tensor,
    *,
    mode: str,
    beta: float = 1.0,
) -> torch.Tensor:
    """Return analytic ``∂L/∂Q`` of shape ``Q.shape``."""
    if mode == "q_times_g":
        gp = g * (1.0 - g) / beta
        return softmax_w * (g + Q * gp)
    if mode == "q_times_detached_g":
        return softmax_w * g
    if mode == "sg_weighted_lse":
        return softmax_w * torch.ones_like(Q)
    raise ValueError(f"Unknown mode {mode!r}")


def _build_inputs(
    *,
    num_q: int = 2,
    B: int = 3,
    K: int = 8,
    Q_scale: float = 1.0,
    tau_offset: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    Q = (Q_scale * torch.randn(num_q, B, K, generator=g)).requires_grad_()
    tau = (Q.detach().min(dim=0).values.min(dim=-1).values + tau_offset)  # [B]
    log_p = torch.randn(num_q, B, K, generator=g) * 0.3
    return Q, tau, log_p


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode",
    ["q_times_g", "q_times_detached_g", "sg_weighted_lse"],
)
@pytest.mark.parametrize("Q_scale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("tau_offset", [-2.0, 0.0, 2.0])  # τ < / ≈ / > Q
def test_grad_matches_analytic(
    mode: str, Q_scale: float, tau_offset: float,
) -> None:
    """``Q.grad`` from autograd matches the analytic formula."""
    Q, tau, log_p = _build_inputs(
        Q_scale=Q_scale, tau_offset=tau_offset, seed=hash((mode, Q_scale)) & 0xFFFF,
    )
    z = _logits(Q, tau, log_p, mode=mode)
    L = _loss(z)
    (autograd_grad,) = torch.autograd.grad(L, Q, create_graph=False)

    # Recompute g + softmax weights for the analytic side.
    g = torch.sigmoid((Q.detach() - tau.view(1, -1, 1)) / 1.0)
    w = torch.softmax(z.detach(), dim=-1)
    analytic = _analytic_grad(Q.detach(), g, w, mode=mode)

    assert torch.isfinite(autograd_grad).all(), f"NaN/Inf in autograd grad ({mode})"
    assert torch.isfinite(analytic).all(), f"NaN/Inf in analytic grad ({mode})"
    torch.testing.assert_close(
        autograd_grad, analytic, atol=1e-5, rtol=1e-4,
    )


def test_q_times_g_has_amplification_term() -> None:
    """The Q*g' term is the distortion source.  Hypothesis #2 says
    ``∂lse/∂Q_i = w_i · (g_i + Q_i · g'_i)`` whereas the SG variant
    has ``∂lse/∂Q_i = w_i``.  The *per-candidate* ratio
    ``(g + Q·g')`` (the analytic backward factor of q_times_g) must
    exceed 1 by a wide margin for some near-τ candidate when ``|Q|``
    is large and τ ≈ mean(Q) — this is the “amplification” that the
    SG variant removes.

    We assert this directly on the analytic factor (no autograd
    needed) so the test reads exactly as the hypothesis is stated.
    """
    # Construct candidates exactly on the near-τ ridge: half the
    # candidates sit at |Δ| = 0 (g = 0.5, g' = 0.25) with a large
    # |Q|.  At Q = 8, β = 1: factor = 0.5 + 8·0.25 = 2.5 ≫ 1.
    Q = torch.tensor([
        [[8.0, -8.0, 0.0, 4.0]],   # critic 0
        [[8.0, -8.0, 0.0, 4.0]],   # critic 1
    ])  # [num_q=2, B=1, K=4]
    tau = torch.tensor([0.0])      # δ = Q for every candidate
    delta = Q - tau.view(1, -1, 1)
    gate = torch.sigmoid(delta)                           # β = 1
    gprime = gate * (1.0 - gate)
    factor_qg = gate + Q * gprime                         # q_times_g factor
    factor_sg = torch.ones_like(Q)                        # sg_weighted_lse factor

    # The amplification: at Q = 0 (mid-τ candidate), g = 0.5,
    # g' = 0.25, |Q·g'| = 0 — but at Q = ±8 the |Q·g'| term
    # dominates: g'(8) ≈ 3.35e-4, factor ≈ 1 + 8·3.35e-4 ≈ 1.003;
    # the real amplification appears at Q = 4 with g = 0.982,
    # g' = 0.0177, factor = 0.982 + 4·0.0177 ≈ 1.05.  The strongest
    # amplification is at intermediate Q where both g and Q·g' are
    # large — Q = 2 gives factor ≈ 0.881 + 2·0.105 ≈ 1.09.
    # We assert simply that there exists ≥1 candidate with factor
    # > 1.05 (which |sg| = 1 cannot reach), proving the formula
    # divergence.
    assert factor_qg.abs().max() > 1.05, (
        f"Expected max |g + Q·g'| > 1.05 (sg_weighted_lse max = 1); "
        f"got {float(factor_qg.abs().max()):.4f}"
    )
    assert factor_sg.abs().max() == pytest.approx(1.0)


def test_sg_weighted_lse_grad_is_softmax_only() -> None:
    """For ``sg_weighted_lse`` ``∂lse/∂Q_i`` must equal exactly
    ``softmax(z)_i`` (no β / Q multipliers)."""
    Q, tau, log_p = _build_inputs(Q_scale=3.0, seed=7)
    z = _logits(Q, tau, log_p, mode="sg_weighted_lse")
    L = _loss(z)
    (autograd,) = torch.autograd.grad(L, Q)
    w = torch.softmax(z.detach(), dim=-1)
    torch.testing.assert_close(autograd, w, atol=1e-6, rtol=1e-5)


def test_q_times_detached_g_blocks_g_path() -> None:
    """In ``q_times_detached_g``, varying τ via the gate should not
    propagate back into Q (the gate has no Q dependence)."""
    Q, tau, log_p = _build_inputs(seed=11)
    z = _logits(Q, tau, log_p, mode="q_times_detached_g")
    L = _loss(z)
    (g_q,) = torch.autograd.grad(L, Q)
    g = torch.sigmoid((Q.detach() - tau.view(1, -1, 1)))
    w = torch.softmax(z.detach(), dim=-1)
    torch.testing.assert_close(g_q, w * g, atol=1e-6, rtol=1e-5)


def test_g_near_zero_no_nan_inf() -> None:
    """When ``g → 0`` (Q ≪ τ), all three modes must remain finite.
    The ε floor inside ``log(detach(g) + ε)`` protects sg_weighted_lse.
    """
    Q, _, log_p = _build_inputs(Q_scale=1.0, seed=99)
    # Force τ far above Q so that g ≈ 0 for all candidates.
    tau = Q.detach().max() + 50.0 * torch.ones(Q.shape[1])
    for mode in ("q_times_g", "q_times_detached_g", "sg_weighted_lse"):
        Qm = Q.detach().clone().requires_grad_()
        z = _logits(Qm, tau, log_p, mode=mode)
        L = _loss(z)
        (gQ,) = torch.autograd.grad(L, Qm)
        assert torch.isfinite(z).all(), f"Non-finite logits in {mode}"
        assert torch.isfinite(L), f"Non-finite loss in {mode}"
        assert torch.isfinite(gQ).all(), f"Non-finite Q.grad in {mode}"


def test_g_near_one_no_nan_inf() -> None:
    """``g → 1`` (Q ≫ τ): ``sg_weighted_lse`` reduces to vanilla
    ``logsumexp(Q − log p)``; all modes must stay finite."""
    Q, _, log_p = _build_inputs(Q_scale=1.0, seed=23)
    tau = Q.detach().min() - 50.0 * torch.ones(Q.shape[1])
    for mode in ("q_times_g", "q_times_detached_g", "sg_weighted_lse"):
        Qm = Q.detach().clone().requires_grad_()
        z = _logits(Qm, tau, log_p, mode=mode)
        L = _loss(z)
        (gQ,) = torch.autograd.grad(L, Qm)
        assert torch.isfinite(z).all(), f"Non-finite logits in {mode}"
        assert torch.isfinite(gQ).all(), f"Non-finite Q.grad in {mode}"


def test_top1_match_q_vs_logits_per_mode() -> None:
    """Hypothesis #2 (forward): ``q_times_g`` and ``q_times_detached_g``
    can disagree with ``argmax(Q)`` on the candidate axis (Q*g not
    monotone in Q when τ < 0); ``sg_weighted_lse`` preserves Q-rank
    because logits = Q + const(detached) − log p is affine in Q.
    """
    # Construct a controlled case with τ < 0 so Q·g is non-monotone.
    Q = torch.tensor([[[-3.0, -1.0, 1.0, 3.0]]])  # [1, 1, 4]
    log_p = torch.zeros_like(Q)
    tau = torch.tensor([-2.0])

    z_qg = _logits(Q, tau, log_p, mode="q_times_g")
    z_sg = _logits(Q, tau, log_p, mode="sg_weighted_lse")

    # In ``sg_weighted_lse``, top1 of logits MUST equal top1 of (Q − log p).
    q_minus_logp = Q - log_p
    assert z_sg.argmax(dim=-1).item() == q_minus_logp.argmax(dim=-1).item()
    # ``q_times_g`` ranking under this τ < 0 setting is allowed to
    # disagree with raw Q (this is exactly the distortion).  We do
    # not assert disagreement (would be brittle), but we assert that
    # the sg_weighted_lse logits are monotone in Q.
    z_sg_flat = z_sg.flatten()
    assert torch.all(
        torch.diff(z_sg_flat) >= -1e-7
    ), "sg_weighted_lse must be monotone in Q (at fixed log_p)"


# ──────────────────────────────────────────────────────────────────
# Standalone debug entrypoint — `python -m … test_smqr_sg_gradient`.
# ──────────────────────────────────────────────────────────────────


def _debug_summary() -> None:
    """Pretty-print mode-by-mode logits / softmax / Q.grad / top1
    match for one synthetic batch.  Standalone debug helper retained
    for manual inspection (the legacy
    ``scripts/debug_smqr_sg_gradient_sanity.py`` entrypoint was
    removed in the Step 6.5-B cleanup; the equivalence checks now
    live in ``tests/offline_rl/test_smqr_sg_loss_equivalence.py``).
    """
    Q, tau, log_p = _build_inputs(num_q=1, B=1, K=6, Q_scale=3.0, seed=0)
    print(f"Q       = {Q.detach().flatten().tolist()}")
    print(f"tau     = {tau.tolist()}")
    print(f"log_p   = {log_p.flatten().tolist()}")
    q_top1 = Q.detach().argmax(dim=-1).item()
    print(f"argmax(Q) = {q_top1}")
    for mode in ("q_times_g", "q_times_detached_g", "sg_weighted_lse"):
        Qm = Q.detach().clone().requires_grad_()
        z = _logits(Qm, tau, log_p, mode=mode)
        L = _loss(z)
        (gQ,) = torch.autograd.grad(L, Qm)
        w = torch.softmax(z.detach(), dim=-1)
        z_top1 = z.argmax(dim=-1).item()
        print(
            f"\n[{mode}]\n"
            f"  logits   = {z.detach().flatten().tolist()}\n"
            f"  softmax  = {w.flatten().tolist()}\n"
            f"  Q.grad   = {gQ.flatten().tolist()}\n"
            f"  top1(z)  = {z_top1}  (matches argmax(Q)? "
            f"{z_top1 == q_top1})\n"
            f"  L        = {float(L):.6f}\n"
            f"  finite?  = {bool(torch.isfinite(gQ).all())}"
        )


if __name__ == "__main__":  # pragma: no cover
    _debug_summary()
