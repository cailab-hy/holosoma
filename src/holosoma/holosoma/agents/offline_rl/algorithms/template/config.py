"""Template algorithm config placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemplateAlgorithmConfig:
    """Minimal placeholder.  Replace with real fields when implementing.

    DO NOT subclass :class:`OfflineCQLConfig` in production code without
    auditing field collisions; the legacy config has ~200 fields and
    most are CQL/SMQR specific.  For a clean baseline, declare only
    the fields the new algorithm actually needs.
    """

    name: str = "template"
    placeholder: bool = True
