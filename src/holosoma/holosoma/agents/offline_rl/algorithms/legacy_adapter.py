"""Offline-RL train-command adapter.

Read-only utilities that map registry algorithm keys to the artefacts
needed by the historical offline-RL train preset
(``src/holosoma/holosoma/train_agent.py`` + direct canonical agents).

Design contract
---------------
This module DOES NOT:

* instantiate agents (it never imports torch);
* run training;
* alter ``train_agent.py`` / ``eval_agent.py`` / production replication scripts.

This module DOES:

* expose ``LegacyAlgorithmAdapter`` — a tiny dataclass that bundles
  algorithm metadata into a form the opt-in runner can consume;
* convert ``snake_case`` config keys to the ``--algo.config.kebab-case``
  CLI flag form used by ``train_agent.py``;
* build the canonical Python invocation list that the historical train
  path expects (preset + ``--algo.config.*`` flags + ``--training.*``).

It is acceptable for callers to build adapters either from the
registry (the common path) or directly (for ad-hoc tooling).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from holosoma.agents.offline_rl.algorithms.registry import (
    AlgorithmEntry,
    get_algorithm_entry,
)
from holosoma.agents.offline_rl.common.target_compat import LEGACY_TARGET
from holosoma.agents.offline_rl.datasets.registry import (
    DatasetEntry,
    get_dataset_entry,
)


# Algorithms exposed by the historical offline-RL train preset.
_LEGACY_FAMILIES: frozenset[str] = frozenset({"cql", "smqr", "smqr_sg"})

# Canonical preset string for the WBT-object dataset (matches
# scripts/train_replication/_common.sh ``PRESET``).
_WBT_OBJECT_PRESET = "exp:g1-29dof-wbt-offline-cql-w-object"

# Mapping from dataset key → preset string.  Extensible later; for
# now there is only one dataset family that the legacy agent supports.
_DATASET_PRESETS: Mapping[str, str] = {
    "wbt_object": _WBT_OBJECT_PRESET,
}


def _to_cli_flag(key: str) -> str:
    """Convert ``critic_penalty_mode`` → ``--algo.config.critic-penalty-mode``."""
    return f"--algo.config.{key.replace('_', '-')}"


def _stringify(value: Any) -> str:
    """Render a config value in CLI form.

    The legacy ``train_agent.py`` parses bools via ``tyro``, which
    expects Python-style ``True`` / ``False`` (capitalised).
    ``int`` / ``float`` use the standard ``repr`` form; strings pass
    through verbatim (callers quote later if needed).
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


@dataclass(frozen=True)
class LegacyAlgorithmAdapter:
    """Read-only bundle describing how to launch a legacy algorithm.

    Fields
    ------
    algorithm
        The :class:`AlgorithmEntry` looked up from the registry.
    dataset
        The :class:`DatasetEntry` looked up from the registry.
    preset
        ``train_agent.py`` positional preset string (e.g.
        ``"exp:g1-29dof-wbt-offline-cql-w-object"``).
    legacy_agent_class_path
        Historical dotted path kept as metadata only; identical to
        :data:`holosoma.agents.offline_rl.common.target_compat.LEGACY_TARGET`.
        The runner does not import it; old target compatibility is handled by
        ``target_compat``.
    """

    algorithm: AlgorithmEntry
    dataset: DatasetEntry
    preset: str
    legacy_agent_class_path: str

    # ── Public helpers ───────────────────────────────────────────
    def algo_config_flags(
        self,
        *,
        overrides: Mapping[str, Any] | None = None,
        include_optional: bool = True,
    ) -> list[str]:
        """Render ``--algo.config.<kebab-key> <value>`` pairs.

        Parameters
        ----------
        overrides
            Extra config knobs to forward (e.g. parsed from a YAML
            ``hyperparameters`` block).  Keys here take precedence
            over the algorithm's documented defaults.
        include_optional
            When ``True`` (default), include all of
            ``legacy_config_keys``.  When ``False``, restrict to the
            ``required_config_keys`` subset.
        """
        keys: Mapping[str, Any] = dict(self.algorithm.legacy_config_keys)
        if not include_optional:
            keys = {
                k: v
                for k, v in keys.items()
                if k in self.algorithm.required_config_keys
            }
        if overrides:
            keys = {**keys, **overrides}
        flags: list[str] = []
        for k, v in keys.items():
            flags.append(_to_cli_flag(k))
            flags.append(_stringify(v))
        return flags

    def build_train_command(
        self,
        *,
        run_tag: str,
        seed: int,
        num_iters: int,
        save_interval: int,
        dataset_path: str = "offline_data/fastsac_dataset.h5",
        python_bin: str = "python",
        train_agent_path: str = "src/holosoma/holosoma/train_agent.py",
        logger: str = "wandb",
        overrides: Mapping[str, Any] | None = None,
        include_optional_config_keys: bool = True,
    ) -> list[str]:
        """Compose the full argv that reproduces the legacy launch.

        The shape matches ``scripts/train_replication/_common.sh ::
        run_training`` exactly.  Returned as a Python list so callers
        can either ``subprocess.run`` it or print it verbatim.
        """
        cmd: list[str] = [
            python_bin,
            train_agent_path,
            self.preset,
            f"logger:{logger}",
            "--training.seed",
            str(seed),
            "--training.name",
            run_tag,
            "--algo.config.dataset-path",
            dataset_path,
            "--algo.config.num-learning-iterations",
            str(num_iters),
            "--algo.config.save-interval",
            str(save_interval),
        ]
        cmd.extend(
            self.algo_config_flags(
                overrides=overrides,
                include_optional=include_optional_config_keys,
            )
        )
        return cmd


def build_legacy_adapter(
    algorithm_key: str,
    dataset_key: str | None = None,
) -> LegacyAlgorithmAdapter:
    """Look up registries and assemble a :class:`LegacyAlgorithmAdapter`.

    Parameters
    ----------
    algorithm_key
        One of the registered legacy algorithms (``cql`` / ``smqr`` /
        ``smqr_sg``).  Placeholders (``bc``, ``iql``, ``td3_bc``,
        ``awac``) raise ``NotImplementedError`` — the legacy adapter
        only models the legacy train path.
    dataset_key
        Override for the dataset registry key; defaults to the
        algorithm's ``default_dataset_key``.  Placeholders raise
        ``NotImplementedError``.
    """
    algo = get_algorithm_entry(algorithm_key)
    if algo.status != "legacy" or algo.family not in _LEGACY_FAMILIES:
        raise NotImplementedError(
            f"algorithm {algorithm_key!r} (status={algo.status!r}, "
            f"family={algo.family!r}) is not served by the legacy "
            f"train path."
        )

    dkey = dataset_key or algo.default_dataset_key
    if dkey is None:
        raise ValueError(
            f"algorithm {algorithm_key!r} has no default_dataset_key; "
            f"pass dataset_key explicitly."
        )
    dataset = get_dataset_entry(dkey)
    if dataset.status != "legacy":
        raise NotImplementedError(
            f"dataset {dkey!r} (status={dataset.status!r}) is a "
            f"placeholder; legacy adapter only supports legacy datasets."
        )

    preset = _DATASET_PRESETS.get(dkey)
    if preset is None:
        raise KeyError(
            f"no train_agent.py preset registered for dataset {dkey!r}"
        )

    return LegacyAlgorithmAdapter(
        algorithm=algo,
        dataset=dataset,
        preset=preset,
        legacy_agent_class_path=LEGACY_TARGET,
    )


def render_shell_command(argv: Sequence[str]) -> str:
    """Render an argv list as a copy-pasteable shell line."""
    return " ".join(shlex.quote(str(a)) for a in argv)


__all__ = [
    "LegacyAlgorithmAdapter",
    "build_legacy_adapter",
    "render_shell_command",
]
