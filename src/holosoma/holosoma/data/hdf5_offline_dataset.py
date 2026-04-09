from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info


REQUIRED_HDF5_KEYS: tuple[str, ...] = (
    "observations",
    "actions",
    "critic_observations",
    "next_observations",
    "next_critic_observations",
    "rewards",
    "truncations",
    "dones",
)


@dataclass(frozen=True)
class HDF5OfflineRLSpec:
    num_samples: int
    observation_dim: int
    action_dim: int
    critic_observation_dim: int
    bytes_per_transition: int


def _to_numpy_indices(indices: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape={array.shape}")
    return array


def _pin_if_possible(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    if not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str, non_blocking: bool = True) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device=device, non_blocking=non_blocking)
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def batch_to_device(batch: dict[str, Any], device: torch.device | str, non_blocking: bool = True) -> dict[str, Any]:
    return move_batch_to_device(batch, device=device, non_blocking=non_blocking)


def apply_observation_normalization(
    batch: dict[str, Any],
    normalize_obs,
    normalize_critic_obs,
) -> dict[str, Any]:
    batch["observations"] = normalize_obs(batch["observations"])
    batch["critic_observations"] = normalize_critic_obs(batch["critic_observations"])
    batch["next"]["observations"] = normalize_obs(batch["next"]["observations"])
    batch["next"]["critic_observations"] = normalize_critic_obs(batch["next"]["critic_observations"])
    return batch


def _squeeze_sample(batch: dict[str, Any]) -> dict[str, Any]:
    squeezed: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            squeezed[key] = _squeeze_sample(value)
        elif isinstance(value, torch.Tensor):
            squeezed[key] = value[0]
        else:
            squeezed[key] = value
    return squeezed


def _slice_nested_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            sliced[key] = _slice_nested_batch(value, start, end)
        elif isinstance(value, torch.Tensor):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _cat_nested_batches(batches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("batches must not be empty")

    merged: dict[str, Any] = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], dict):
            merged[key] = _cat_nested_batches(values)  # type: ignore[arg-type]
        elif isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = values[-1]
    return merged


def _index_nested_batch(batch: dict[str, Any], indices: torch.Tensor, pin_memory: bool) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            indexed[key] = _index_nested_batch(value, indices, pin_memory=pin_memory)
        elif isinstance(value, torch.Tensor):
            indexed_value = value.index_select(0, indices).contiguous()
            indexed[key] = _pin_if_possible(indexed_value, enabled=pin_memory)
        else:
            indexed[key] = value
    return indexed


class HDF5ReplayReader:
    """Lazy HDF5 reader that only loads requested rows into CPU memory."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_observation_dim: int | None = None,
        expected_action_dim: int | None = None,
        expected_critic_observation_dim: int | None = None,
        pin_memory: bool = True,
        swmr: bool = True,
    ) -> None:
        self.path = Path(path)
        self.expected_observation_dim = expected_observation_dim
        self.expected_action_dim = expected_action_dim
        self.expected_critic_observation_dim = expected_critic_observation_dim
        self.pin_memory = pin_memory
        self.swmr = swmr

        self._file: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._spec: HDF5OfflineRLSpec | None = None

        self._initialize_metadata()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_datasets"] = {}
        return state

    @property
    def spec(self) -> HDF5OfflineRLSpec:
        if self._spec is None:
            raise RuntimeError("Reader metadata is not initialized.")
        return self._spec

    @property
    def num_samples(self) -> int:
        return self.spec.num_samples

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._datasets = {}

    def __del__(self) -> None:
        self.close()

    def _initialize_metadata(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")

        with h5py.File(self.path, "r") as h5_file:
            missing_keys = [key for key in REQUIRED_HDF5_KEYS if key not in h5_file]
            if missing_keys:
                raise KeyError(f"HDF5 dataset is missing required keys: {missing_keys}")

            num_samples = int(h5_file.attrs.get("num_samples", h5_file["observations"].shape[0]))
            if num_samples <= 0:
                raise ValueError(f"HDF5 dataset '{self.path}' has no samples.")

            first_dims = {key: int(h5_file[key].shape[0]) for key in REQUIRED_HDF5_KEYS}
            inconsistent = {key: dim for key, dim in first_dims.items() if dim < num_samples}
            if inconsistent:
                raise ValueError(
                    f"Some datasets are shorter than num_samples={num_samples}: {inconsistent}"
                )

            observation_dim = self._validate_feature_dim(h5_file, "observations", self.expected_observation_dim)
            action_dim = self._validate_feature_dim(h5_file, "actions", self.expected_action_dim)
            critic_observation_dim = self._validate_feature_dim(
                h5_file,
                "critic_observations",
                self.expected_critic_observation_dim,
            )
            self._validate_feature_dim(h5_file, "next_observations", observation_dim)
            self._validate_feature_dim(h5_file, "next_critic_observations", critic_observation_dim)
            self._validate_scalar_dim(h5_file, "rewards")
            self._validate_scalar_dim(h5_file, "truncations")
            self._validate_scalar_dim(h5_file, "dones")

            self._spec = HDF5OfflineRLSpec(
                num_samples=num_samples,
                observation_dim=observation_dim,
                action_dim=action_dim,
                critic_observation_dim=critic_observation_dim,
                bytes_per_transition=self._compute_bytes_per_transition(h5_file),
            )

    def _open_if_needed(self) -> None:
        if self._file is not None:
            return
        try:
            self._file = h5py.File(self.path, "r", libver="latest", swmr=self.swmr)
        except (OSError, ValueError):
            self._file = h5py.File(self.path, "r")
        self._datasets = {key: self._file[key] for key in REQUIRED_HDF5_KEYS}

    @staticmethod
    def _validate_feature_dim(h5_file: h5py.File, key: str, expected_dim: int | None) -> int:
        dataset = h5_file[key]
        if dataset.ndim != 2:
            raise ValueError(f"'{key}' must be rank-2 [N, D], got shape={dataset.shape}")
        dim = int(dataset.shape[1])
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(f"'{key}' has dim={dim}, expected dim={expected_dim}")
        return dim

    @staticmethod
    def _validate_scalar_dim(h5_file: h5py.File, key: str) -> None:
        dataset = h5_file[key]
        if dataset.ndim == 1:
            return
        if dataset.ndim == 2 and dataset.shape[1] == 1:
            return
        raise ValueError(f"'{key}' must have shape [N] or [N, 1], got shape={dataset.shape}")

    @staticmethod
    def _compute_bytes_per_transition(h5_file: h5py.File) -> int:
        total = 0
        for key in REQUIRED_HDF5_KEYS:
            dataset = h5_file[key]
            trailing_shape = dataset.shape[1:] if dataset.ndim > 1 else ()
            num_values = int(np.prod(trailing_shape, dtype=np.int64)) if trailing_shape else 1
            total += num_values * dataset.dtype.itemsize
        return total

    def sample_indices(
        self,
        batch_size: int,
        *,
        replacement: bool | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if rng is None:
            rng = np.random.default_rng()
        if replacement is None:
            replacement = batch_size > self.num_samples
        if replacement:
            return rng.integers(0, self.num_samples, size=batch_size, dtype=np.int64)
        return rng.choice(self.num_samples, size=batch_size, replace=False).astype(np.int64, copy=False)

    def _read_rows(self, key: str, indices: np.ndarray) -> np.ndarray:
        self._open_if_needed()
        dataset = self._datasets[key]
        if indices.size == 0:
            return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)

        sorted_positions = np.argsort(indices, kind="stable")
        sorted_indices = indices[sorted_positions]

        chunks: list[np.ndarray] = []
        start = 0
        while start < sorted_indices.size:
            stop = start + 1
            while stop < sorted_indices.size and sorted_indices[stop] == sorted_indices[stop - 1] + 1:
                stop += 1
            first_idx = int(sorted_indices[start])
            last_idx = int(sorted_indices[stop - 1])
            chunks.append(np.asarray(dataset[first_idx : last_idx + 1]))
            start = stop

        sorted_rows = np.concatenate(chunks, axis=0)
        inverse_positions = np.empty_like(sorted_positions)
        inverse_positions[sorted_positions] = np.arange(sorted_positions.size)
        return np.ascontiguousarray(sorted_rows[inverse_positions])

    def read_batch(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> dict[str, Any]:
        row_indices = _to_numpy_indices(indices)
        batch_size = int(row_indices.shape[0])

        observations = torch.from_numpy(self._read_rows("observations", row_indices)).to(torch.float32)
        actions = torch.from_numpy(self._read_rows("actions", row_indices)).to(torch.float32)
        critic_observations = torch.from_numpy(self._read_rows("critic_observations", row_indices)).to(torch.float32)
        next_observations = torch.from_numpy(self._read_rows("next_observations", row_indices)).to(torch.float32)
        next_critic_observations = torch.from_numpy(self._read_rows("next_critic_observations", row_indices)).to(
            torch.float32
        )
        rewards = torch.from_numpy(self._read_rows("rewards", row_indices)).to(torch.float32).reshape(batch_size)
        truncations = torch.from_numpy(self._read_rows("truncations", row_indices)).to(torch.bool).reshape(batch_size)
        dones = torch.from_numpy(self._read_rows("dones", row_indices)).to(torch.bool).reshape(batch_size)

        batch = {
            "observations": _pin_if_possible(observations.contiguous(), enabled=self.pin_memory),
            "actions": _pin_if_possible(actions.contiguous(), enabled=self.pin_memory),
            "critic_observations": _pin_if_possible(critic_observations.contiguous(), enabled=self.pin_memory),
            "next": {
                "observations": _pin_if_possible(next_observations.contiguous(), enabled=self.pin_memory),
                "critic_observations": _pin_if_possible(next_critic_observations.contiguous(), enabled=self.pin_memory),
                "rewards": _pin_if_possible(rewards.contiguous(), enabled=self.pin_memory),
                "truncations": _pin_if_possible(truncations.contiguous(), enabled=self.pin_memory),
                "dones": _pin_if_possible(dones.contiguous(), enabled=self.pin_memory),
                "effective_n_steps": _pin_if_possible(
                    torch.ones(batch_size, dtype=torch.long),
                    enabled=self.pin_memory,
                ),
            },
        }

        if device is not None:
            batch = move_batch_to_device(batch, device=device, non_blocking=non_blocking)
        return batch

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        replacement: bool | None = None,
        rng: np.random.Generator | None = None,
        non_blocking: bool = True,
    ) -> dict[str, Any]:
        indices = self.sample_indices(batch_size, replacement=replacement, rng=rng)
        return self.read_batch(indices, device=device, non_blocking=non_blocking)


class HDF5BlockReader(HDF5ReplayReader):
    """Read large contiguous blocks from HDF5 for CPU-side shuffle buffering."""

    def read_block(
        self,
        start: int,
        block_size: int,
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> dict[str, Any]:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if start < 0 or start >= self.num_samples:
            raise IndexError(f"block start {start} is out of range for dataset of size {self.num_samples}")

        end = min(start + block_size, self.num_samples)
        self._open_if_needed()
        block_length = end - start
        if block_length <= 0:
            raise ValueError(f"Requested empty block [{start}:{end}]")

        observations = torch.from_numpy(np.asarray(self._datasets["observations"][start:end])).to(torch.float32)
        actions = torch.from_numpy(np.asarray(self._datasets["actions"][start:end])).to(torch.float32)
        critic_observations = torch.from_numpy(np.asarray(self._datasets["critic_observations"][start:end])).to(
            torch.float32
        )
        next_observations = torch.from_numpy(np.asarray(self._datasets["next_observations"][start:end])).to(
            torch.float32
        )
        next_critic_observations = torch.from_numpy(np.asarray(self._datasets["next_critic_observations"][start:end])).to(
            torch.float32
        )
        rewards = torch.from_numpy(np.asarray(self._datasets["rewards"][start:end])).to(torch.float32).reshape(block_length)
        truncations = torch.from_numpy(np.asarray(self._datasets["truncations"][start:end])).to(torch.bool).reshape(
            block_length
        )
        dones = torch.from_numpy(np.asarray(self._datasets["dones"][start:end])).to(torch.bool).reshape(block_length)

        block = {
            "observations": observations.contiguous(),
            "actions": actions.contiguous(),
            "critic_observations": critic_observations.contiguous(),
            "next": {
                "observations": next_observations.contiguous(),
                "critic_observations": next_critic_observations.contiguous(),
                "rewards": rewards.contiguous(),
                "truncations": truncations.contiguous(),
                "dones": dones.contiguous(),
                "effective_n_steps": torch.ones(block_length, dtype=torch.long),
            },
        }
        if device is not None:
            block = move_batch_to_device(block, device=device, non_blocking=non_blocking)
        return block

    def iter_block_slices(
        self,
        block_size: int,
        *,
        shuffle: bool,
        rng: np.random.Generator,
    ) -> Iterator[tuple[int, int]]:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        starts = np.arange(0, self.num_samples, block_size, dtype=np.int64)
        if shuffle:
            rng.shuffle(starts)
        for start in starts.tolist():
            end = min(start + block_size, self.num_samples)
            yield start, end


class RAMShuffleBuffer:
    """CPU-side bounded shuffle buffer fed by contiguous HDF5 block reads."""

    def __init__(
        self,
        block_reader: HDF5BlockReader,
        *,
        block_size: int,
        capacity: int,
        refill_threshold: int,
        pin_memory: bool = True,
        shuffle_block_order: bool = True,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > capacity:
            raise ValueError(f"block_size={block_size} must be <= capacity={capacity}")
        if refill_threshold < 0:
            raise ValueError(f"refill_threshold must be >= 0, got {refill_threshold}")
        if refill_threshold >= capacity:
            raise ValueError(
                f"refill_threshold={refill_threshold} must be smaller than capacity={capacity}"
            )

        self.block_reader = block_reader
        self.block_size = block_size
        self.capacity = capacity
        self.refill_threshold = refill_threshold
        self.pin_memory = pin_memory
        self.shuffle_block_order = shuffle_block_order
        self.rng = np.random.default_rng(seed)

        self._storage: dict[str, Any] | None = None
        self._num_items = 0
        self._sample_order = np.empty(0, dtype=np.int64)
        self._cursor = 0
        self._pending_block: tuple[int, int] | None = None
        self._block_iterator = self.block_reader.iter_block_slices(
            self.block_size,
            shuffle=self.shuffle_block_order,
            rng=self.rng,
        )

    @property
    def num_items(self) -> int:
        return self._num_items

    @property
    def remaining(self) -> int:
        return max(self._num_items - self._cursor, 0)

    @property
    def capacity_bytes(self) -> int:
        return self.capacity * self.block_reader.spec.bytes_per_transition

    def close(self) -> None:
        self.block_reader.close()

    def __del__(self) -> None:
        self.close()

    def _next_block_bounds(self) -> tuple[int, int]:
        try:
            return next(self._block_iterator)
        except StopIteration:
            self._block_iterator = self.block_reader.iter_block_slices(
                self.block_size,
                shuffle=self.shuffle_block_order,
                rng=self.rng,
            )
            return next(self._block_iterator)

    def _take_next_read_slice(self, max_size: int) -> tuple[int, int]:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        if self._pending_block is None:
            self._pending_block = self._next_block_bounds()

        start, end = self._pending_block
        read_end = min(start + max_size, end)
        if read_end >= end:
            self._pending_block = None
        else:
            self._pending_block = (read_end, end)
        return start, read_end

    def _compact_remaining(self) -> None:
        if self._storage is None:
            self._num_items = 0
            self._sample_order = np.empty(0, dtype=np.int64)
            self._cursor = 0
            return

        remaining = self.remaining
        if remaining == self._num_items and self._cursor == 0:
            return
        if remaining == 0:
            self._storage = None
            self._num_items = 0
            self._sample_order = np.empty(0, dtype=np.int64)
            self._cursor = 0
            return

        keep_indices = torch.from_numpy(self._sample_order[self._cursor : self._cursor + remaining].copy()).to(torch.long)
        self._storage = _index_nested_batch(self._storage, keep_indices, pin_memory=False)
        self._num_items = remaining
        self._sample_order = np.empty(0, dtype=np.int64)
        self._cursor = 0

    def _append_block(self, block: dict[str, Any]) -> None:
        block_len = int(block["observations"].shape[0])
        if block_len == 0:
            return
        if self._num_items + block_len > self.capacity:
            raise ValueError(
                f"Appending block of size {block_len} would exceed shuffle buffer capacity {self.capacity}"
            )

        if self._storage is None:
            self._storage = block
        else:
            self._storage = _cat_nested_batches([self._storage, block])
        self._num_items += block_len

    def refill(self, min_required: int = 0) -> None:
        if min_required > self.capacity:
            raise ValueError(
                f"min_required={min_required} cannot exceed shuffle buffer capacity={self.capacity}"
            )

        self._compact_remaining()
        target_size = self.capacity
        while self._num_items < target_size and self._num_items < self.capacity:
            available = self.capacity - self._num_items
            read_size = min(self.block_size, available)
            start, end = self._take_next_read_slice(read_size)
            block = self.block_reader.read_block(start=start, block_size=end - start)
            self._append_block(block)

        if self._num_items == 0:
            raise RuntimeError("Shuffle buffer is empty after refill.")

        self._sample_order = self.rng.permutation(self._num_items)
        self._cursor = 0

    def sample(self, batch_size: int) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self.capacity:
            raise ValueError(f"batch_size={batch_size} must be <= shuffle buffer capacity={self.capacity}")

        if self._storage is None or self.remaining < batch_size:
            self.refill(min_required=max(batch_size, self.refill_threshold + 1))
        elif self.remaining <= self.refill_threshold:
            self.refill(min_required=max(batch_size, self.refill_threshold + 1))

        assert self._storage is not None
        sample_indices_np = self._sample_order[self._cursor : self._cursor + batch_size]
        if sample_indices_np.shape[0] < batch_size:
            self.refill(min_required=max(batch_size, self.refill_threshold + 1))
            assert self._storage is not None
            sample_indices_np = self._sample_order[self._cursor : self._cursor + batch_size]
            if sample_indices_np.shape[0] < batch_size:
                raise RuntimeError("Failed to sample a full batch from RAM shuffle buffer.")

        sample_indices = torch.from_numpy(sample_indices_np.copy()).to(torch.long)
        self._cursor += batch_size
        return _index_nested_batch(self._storage, sample_indices, pin_memory=self.pin_memory)


class HDF5TransitionDataset(Dataset[dict[str, Any]]):
    """Map-style dataset for random single-transition access."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_observation_dim: int | None = None,
        expected_action_dim: int | None = None,
        expected_critic_observation_dim: int | None = None,
        pin_memory: bool = False,
    ) -> None:
        self._reader = HDF5ReplayReader(
            path,
            expected_observation_dim=expected_observation_dim,
            expected_action_dim=expected_action_dim,
            expected_critic_observation_dim=expected_critic_observation_dim,
            pin_memory=pin_memory,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader"] = self._reader.__class__(  # type: ignore[call-arg]
            self._reader.path,
            expected_observation_dim=self._reader.expected_observation_dim,
            expected_action_dim=self._reader.expected_action_dim,
            expected_critic_observation_dim=self._reader.expected_critic_observation_dim,
            pin_memory=self._reader.pin_memory,
            swmr=self._reader.swmr,
        )
        return state

    def __len__(self) -> int:
        return self._reader.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _squeeze_sample(self._reader.read_batch([int(index)]))


class RandomBatchSampler(Sampler[list[int]]):
    """Yields batches of random indices without loading the HDF5 file itself."""

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        num_batches: int,
        *,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.replacement = replacement
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_batches):
            if self.replacement:
                indices = rng.integers(0, self.num_samples, size=self.batch_size, dtype=np.int64)
            else:
                indices = rng.choice(self.num_samples, size=self.batch_size, replace=False).astype(
                    np.int64,
                    copy=False,
                )
            yield indices.tolist()

    def __len__(self) -> int:
        return self.num_batches


class HDF5BatchIterableDataset(IterableDataset[dict[str, Any]]):
    """IterableDataset that yields already-batched CPU tensors from HDF5."""

    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int,
        num_batches: int,
        expected_observation_dim: int | None = None,
        expected_action_dim: int | None = None,
        expected_critic_observation_dim: int | None = None,
        replacement: bool = True,
        seed: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.expected_observation_dim = expected_observation_dim
        self.expected_action_dim = expected_action_dim
        self.expected_critic_observation_dim = expected_critic_observation_dim
        self.replacement = replacement
        self.seed = seed
        self.pin_memory = pin_memory

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        reader = HDF5ReplayReader(
            self.path,
            expected_observation_dim=self.expected_observation_dim,
            expected_action_dim=self.expected_action_dim,
            expected_critic_observation_dim=self.expected_critic_observation_dim,
            pin_memory=self.pin_memory,
        )

        batches_per_worker = self.num_batches // num_workers
        if worker_id < self.num_batches % num_workers:
            batches_per_worker += 1

        rng = np.random.default_rng(self.seed + worker_id)
        for _ in range(batches_per_worker):
            yield reader.sample_batch(
                self.batch_size,
                device=None,
                replacement=self.replacement,
                rng=rng,
            )
        reader.close()


def make_hdf5_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    num_batches: int,
    expected_observation_dim: int | None = None,
    expected_action_dim: int | None = None,
    expected_critic_observation_dim: int | None = None,
    replacement: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, Any]]:
    dataset = HDF5BatchIterableDataset(
        path,
        batch_size=batch_size,
        num_batches=num_batches,
        expected_observation_dim=expected_observation_dim,
        expected_action_dim=expected_action_dim,
        expected_critic_observation_dim=expected_critic_observation_dim,
        replacement=replacement,
        seed=seed,
        pin_memory=pin_memory,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True if persistent_workers is None else persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)
