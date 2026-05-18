"""SMQR-SG blend schedules — Step 4-C extraction.

Pure-Python helper that resolves λ(t) for the four schedules currently
supported by ``offline_cql_agent._update_critic`` (see L2430-2459 of
the agent).  No tensor ops; deterministic; trivially unit-testable.

Schedules
---------
* ``fixed``
    ``λ(t) ≡ lambda_start``   (``lambda_end`` ignored).
* ``linear``
    ``frac = clamp(t / ramp_steps, 0, 1)``,
    ``λ(t) = λ_s + (λ_e − λ_s) · frac``.
* ``delayed_linear``
    ``t < warmup_steps``       → ``λ_s``
    ``t ≥ warmup_steps``       → ``λ_s + (λ_e − λ_s) ·
                                  clamp((t − wp) / rp, 0, 1)``
* ``piecewise``
    ``t < wp``                 → ``λ_s``
    ``wp ≤ t < wp + rp``       → linear ramp ``λ_s → λ_e``
    ``t ≥ wp + rp``            → ``λ_e``
    (``hold_steps`` is informational only.)

Bit-exact match: this function reproduces the agent's inline
arithmetic — including the off-by-one boundaries (``<`` vs ``≤``) and
the ``min(max(..., 0.0), 1.0)`` clamp ordering — so the S6 golden
``sg_blend`` + ``fixed`` λ=0.5 config produces identical λ to the
legacy code path.
"""

from __future__ import annotations


def compute_smqr_blend_lambda(
    step: int,
    *,
    schedule: str,
    lambda_start: float,
    lambda_end: float,
    warmup_steps: int = 0,
    ramp_steps: int = 1,
    hold_steps: int | None = None,
) -> float:
    """Resolve λ(t) for the SMQR-SG ``sg_blend`` LOSS-level mix.

    Parameters
    ----------
    step
        ``self.global_step`` at the call site.  Cast to ``int`` here
        to mirror the agent's ``int(self.global_step)``.
    schedule
        One of ``{"fixed", "linear", "delayed_linear", "piecewise"}``.
    lambda_start, lambda_end
        Endpoints of the ramp.  For ``fixed`` only ``lambda_start`` is
        consulted.
    warmup_steps
        Steps to hold at ``lambda_start`` before the ramp begins.
        Only consulted by ``delayed_linear`` and ``piecewise``.
    ramp_steps
        Length of the linear interpolation window.  Must be ≥ 1
        (the agent's setup() validates this; we trust the caller).
    hold_steps
        Informational only; preserved for signature parity with the
        agent's config block.  ``λ`` remains at ``lambda_end`` past
        the ramp regardless.

    Returns
    -------
    float
        ``λ(t)`` cast to Python ``float``.
    """
    _gs = int(step)
    _ls = float(lambda_start)
    _le = float(lambda_end)
    _wp = int(warmup_steps)
    _rp = int(ramp_steps)

    if schedule == "fixed":
        _lam = _ls
    elif schedule == "linear":
        # Bit-exact mirror of agent L2433: ``min(max(gs/rp, 0), 1)``.
        _frac = min(max(_gs / float(_rp), 0.0), 1.0)
        _lam = _ls + (_le - _ls) * _frac
    elif schedule == "delayed_linear":
        if _gs < _wp:
            _lam = _ls
        else:
            _frac = min(
                max((_gs - _wp) / float(_rp), 0.0), 1.0,
            )
            _lam = _ls + (_le - _ls) * _frac
    elif schedule == "piecewise":
        if _gs < _wp:
            _lam = _ls
        elif _gs < _wp + _rp:
            # NOTE: agent uses raw fraction (no clamp) here; the
            # bracketing ``< _wp + _rp`` guarantees 0 ≤ frac < 1.
            _frac = (_gs - _wp) / float(_rp)
            _lam = _ls + (_le - _ls) * _frac
        else:
            _lam = _le
    else:
        raise RuntimeError(f"unknown smqr_blend_schedule={schedule!r}")

    return float(_lam)


__all__ = ["compute_smqr_blend_lambda"]
