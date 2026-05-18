# Adding a new offline-RL baseline algorithm

This directory is a scaffold for new baseline algorithms (BC, IQL,
TD3+BC, AWAC, etc.).  Copy this `template/` package to
`algorithms/<your_algo>/`, rename, and follow the checklist below.

The scaffold deliberately raises `NotImplementedError` everywhere —
you cannot accidentally activate a half-built baseline.

---

## 1. Config

* Declare a frozen dataclass `<YourAlgo>Config` in `config.py`.
* Keep fields minimal — only what the algorithm actually needs.
* Do **NOT** edit `holosoma/config_types/algo.py` (legacy
  `OfflineCQLConfig`) unless you are deliberately migrating an
  existing field.  See `docs/offline_rl_refactor_plan.md` §6 for the
  config-migration policy.
* If you need legacy fields (e.g. `dataset_path`, optimiser
  hyper-params), import them from the legacy config rather than
  redeclaring.

## 2. Loss function

* Add pure-function helpers in `losses.py` following the Step 4
  pattern (see `algorithms/cql/losses.py`,
  `algorithms/smqr/losses.py`,
  `algorithms/smqr_sg/losses.py`):
  - **keyword-only** inputs to prevent accidental positional
    coupling.
  - Return a `TypedDict` of all intermediate tensors so equivalence
    tests can validate every stage.
  - Include a docstring explaining each formula step.
* If your baseline shares math with CQL/SMQR, **import** the existing
  helper rather than copy-pasting.
* No in-place ops on inputs.

## 3. Agent class

* Subclass `BaseAlgo` (or whatever the canonical base becomes after
  the registry/agent integration phase).
* Implement at minimum:
  - `_update_critic(batch) -> dict[str, Tensor]`
  - `_update_actor(batch) -> dict[str, Tensor]`
  - `learn()` — main training loop (or inherit a shared one).
  - `save(path)` / `load(path)` — checkpoint helpers.
* All Tensor metrics returned from `_update_*` must be **scalar**
  (`v.dim() == 0`) and **finite**.  These contracts are enforced by
  the existing `TestUpdateCritic` / `TestUpdateActor` suites.

## 4. Algorithm registry

Add an entry to `algorithms/registry.py`:

```python
_register(AlgorithmEntry(
    name="your_algo",
    status="legacy"  # or "placeholder" during development
    description="One-line summary.",
    helper_modules=(
        "holosoma.agents.offline_rl.algorithms.your_algo.losses",
    ),
    notes="Reference paper / golden run / known limitations.",
))
```

Step-9 note: the previously documented ``legacy_agent_module`` field
was removed. For the canonical legacy dotted path, import
``holosoma.agents.offline_rl.common.target_compat.LEGACY_TARGET``.

Until the registry-driven factory is wired into `train_agent.py`
(future phase), the entry is metadata-only — `train_agent.py` still
dispatches by the legacy config-flag combinations.

## 5. Train script

* Copy an existing replication script
  (e.g. `scripts/train_replication/train_smqr_seed2.sh`) and
  override the relevant `--algo.config.*` flags.
* DO NOT add a brand-new launcher unless `train_agent.py`'s flag
  surface cannot express your new config.

## 6. Eval manifest

* Add the run + checkpoint to `configs/eval/offline_rl_eval_manifest.yaml`
  using the existing `experiment_name / checkpoint_steps` schema.
* The dry-run gate
  (`bash scripts/eval/eval_checkpoints_from_manifest.sh ... --dry-run`)
  must pass before a real eval is run.

## 7. Smoke test

* Author an equivalence test under `tests/offline_rl/` that locks
  the new loss helper against a verbatim inline reference (cf.
  `tests/offline_rl/test_cql_loss_equivalence.py`).
* Add a 50-step runtime smoke harness (model_50.pt + key scalar
  metrics).  No 5K+ training in CI.

## 8. Golden reference

* Once you have a trained checkpoint that meets the acceptance bar
  (success_rate ≥ baseline within tolerance), pin it in
  `configs/golden/offline_rl_golden_manifest.yaml`.
* Update `configs/golden/AUDIT.md` with the audit trail.

## 9. Checkpoint compatibility

* If you introduce a new optimizer or a parameter group not present
  in the legacy SAC/CQL layout, **bump the checkpoint schema
  version** and add a forward-compatibility branch in
  `load_offline_rl_params()`.
* Test load → save → load round-trip in a unit test.

## 10. TensorBoard tag naming

Follow the existing namespace conventions:

* `Loss/<metric>`           — primary loss scalars
* `<algo>/<submetric>`      — algo-specific (e.g. `smqr/blend/...`)
* `actor/<...>`, `critic/<...>` — module-specific

Do **NOT** rename existing tags — the eval pipeline and downstream
notebooks depend on them.
