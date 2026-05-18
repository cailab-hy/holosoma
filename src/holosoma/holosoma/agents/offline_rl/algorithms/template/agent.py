"""Template offline-RL agent placeholder."""

from __future__ import annotations

from typing import Any


class TemplateOfflineAgent:
    """Skeleton class for new baseline agents.

    The real implementation should subclass the relevant ``BaseAlgo``
    abstraction and implement :meth:`_update_actor`,
    :meth:`_update_critic`, :meth:`learn` and the standard checkpoint
    helpers.  See ``README.md`` in this directory for the full
    integration checklist.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "TemplateOfflineAgent is a scaffold only — copy this "
            "package, rename, and replace the placeholders before "
            "instantiating."
        )

    def _update_critic(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _update_actor(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
