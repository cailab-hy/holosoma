"""Template algorithm loss placeholder."""

from __future__ import annotations

from typing import Any


def compute_template_loss(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for the new baseline's loss function.

    Replace with a pure-function helper following the Step 4 pattern:
      * keyword-only inputs
      * returns a TypedDict of tensors
      * bit-exact reference implemented in the equivalence test
    """
    raise NotImplementedError(
        "TemplateOfflineAgent loss is a scaffold only — implement "
        "the new baseline before calling this helper."
    )
