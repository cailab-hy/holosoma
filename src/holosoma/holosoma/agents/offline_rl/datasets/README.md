# Adding a new offline-RL dataset

This directory hosts the dataset registry for offline-RL training.
The current production dataset is **WBT-object**
(`offline_data/fastsac_dataset.h5`); other datasets (locomotion-only,
multi-task, etc.) plug in here.

The scaffold is metadata-only — no dataset I/O is performed in this
package. Production loading goes through the canonical
`holosoma.agents.offline_rl.common.datasets.OfflineDataset` loader.

---

## 1. Dataset file + config

* Place the dataset under `offline_data/<name>.h5` (or
  `offline_data/<name>/` for sharded layouts).
* Record the path, format, and any required environment-side preset
  in `DatasetEntry.legacy_path` and the static `METADATA` dict in
  your dataset module.

## 2. Observation / action spec

* Document the expected observation keys and action dim in
  `DatasetEntry.observation_keys` and `DatasetEntry.action_dim`.
* The OfflineDataset loader expects the H5 to expose the standard
  keys: `obs`, `actions`, `rewards`, `next_obs`, `terminals`.  If
  your dataset has a different schema, write an adapter in the
  dataset module (do NOT modify `OfflineDataset`).

## 3. Normalizer

* If the new dataset has a different scale than WBT-object, you
  must regenerate the frozen normalizer.  See
  `holosoma.agents.offline_rl.common.datasets.create_frozen_normalizer`.
* Pin the normalizer hash in the dataset module's `METADATA` so
  training runs can validate it.

## 4. Replay sampling

* Existing CQL/SMQR/SMQR-SG agents sample uniformly from
  `OfflineDataset`.  No code change required for new datasets that
  use the same sampling strategy.
* If your dataset needs priority sampling, advantage-weighted
  sampling, or sequence sampling, implement it in your dataset
  module — do NOT extend `OfflineDataset`.

## 5. Eval manifest

* Add the new run + checkpoint pattern to
  `configs/eval/offline_rl_eval_manifest.yaml`.
* For a brand-new task, you also need an eval environment preset
  in the existing `experiment_paths.py` system.

## 6. Golden smoke

* Train a small (5K–10K step) reference run and pin the resulting
  checkpoint in `configs/golden/offline_rl_golden_manifest.yaml`.
* Add the audit trail to `configs/golden/AUDIT.md`.

## 7. Train config

* Add an example config to `configs/offline_rl/<algo>_<dataset>.yaml`
  using the existing CQL/SMQR/SMQR-SG example configs as templates.
* The config is example-only until the registry-driven train path is
  wired in a future integration phase.

---

## Registry entry checklist

After implementing the above, register your dataset:

```python
_register(DatasetEntry(
    name="your_dataset",
    status="legacy"  # change from "placeholder" once loader works
    description="One-line summary.",
    legacy_path="offline_data/your_dataset.h5",
    observation_keys=("base_pose", "joint_pos", "..."),
    action_dim=29,
    notes="Reference / origin / known limitations.",
))
```

Step-9 note: ``legacy_loader_module`` was removed. The canonical
loader is ``holosoma.agents.offline_rl.common.datasets.OfflineDataset``
(legacy re-export); reference it directly when documenting a new entry.
