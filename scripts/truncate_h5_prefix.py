#!/usr/bin/env python3
"""Write a smaller Holosoma HDF5 dataset containing only the first prefix fraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an HDF5 dataset while keeping only the first fraction of transition rows."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 dataset path.")
    parser.add_argument("output", type=Path, help="Output HDF5 dataset path.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Prefix fraction to keep. Default: 0.7.",
    )
    parser.add_argument(
        "--num-samples-key",
        default="num_samples",
        help="File attribute key storing the valid transition count. Default: num_samples.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=65536,
        help="Rows copied per chunk for transition datasets. Default: 65536.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _dataset_create_kwargs(src: h5py.Dataset, new_shape: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if src.chunks is not None:
        kwargs["chunks"] = tuple(min(chunk, dim) for chunk, dim in zip(src.chunks, new_shape))
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = True
    if src.fletcher32:
        kwargs["fletcher32"] = True
    if src.scaleoffset is not None:
        kwargs["scaleoffset"] = src.scaleoffset
    return kwargs


def _copy_dataset(
    src: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    *,
    source_num_samples: int,
    keep_count: int,
    chunk_rows: int,
) -> None:
    should_truncate = src.ndim > 0 and int(src.shape[0]) >= source_num_samples
    new_shape = (keep_count, *src.shape[1:]) if should_truncate else src.shape
    dst = dst_parent.create_dataset(name, shape=new_shape, dtype=src.dtype, **_dataset_create_kwargs(src, new_shape))
    _copy_attrs(src.attrs, dst.attrs)

    if src.ndim == 0:
        dst[()] = src[()]
        return

    rows_to_copy = keep_count if should_truncate else int(src.shape[0])
    if rows_to_copy == 0:
        return

    for start in range(0, rows_to_copy, chunk_rows):
        end = min(start + chunk_rows, rows_to_copy)
        dst[start:end] = src[start:end]


def _copy_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    source_num_samples: int,
    keep_count: int,
    chunk_rows: int,
) -> None:
    _copy_attrs(src_group.attrs, dst_group.attrs)
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            _copy_dataset(
                item,
                dst_group,
                name,
                source_num_samples=source_num_samples,
                keep_count=keep_count,
                chunk_rows=chunk_rows,
            )
        elif isinstance(item, h5py.Group):
            child = dst_group.create_group(name)
            _copy_group(
                item,
                child,
                source_num_samples=source_num_samples,
                keep_count=keep_count,
                chunk_rows=chunk_rows,
            )
        else:
            raise TypeError(f"Unsupported HDF5 object at '{item.name}': {type(item)!r}")


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different paths. Refusing in-place truncation.")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")
    if args.chunk_rows <= 0:
        raise ValueError(f"--chunk-rows must be positive, got {args.chunk_rows}")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {args.output}. Use --force to overwrite.")
        args.output.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        if args.num_samples_key in src.attrs:
            source_num_samples = int(src.attrs[args.num_samples_key])
        elif "observations" in src:
            source_num_samples = int(src["observations"].shape[0])
        else:
            raise KeyError(
                f"Could not infer sample count: missing attr '{args.num_samples_key}' and dataset 'observations'."
            )

        keep_count = int(source_num_samples * args.fraction)
        keep_count = max(1, min(keep_count, source_num_samples))

        _copy_group(
            src,
            dst,
            source_num_samples=source_num_samples,
            keep_count=keep_count,
            chunk_rows=args.chunk_rows,
        )
        dst.attrs[args.num_samples_key] = int(keep_count)
        dst.attrs["source_num_samples"] = int(source_num_samples)
        dst.attrs["kept_prefix_fraction"] = float(args.fraction)

    print(
        f"wrote {args.output} | kept_rows={keep_count}/{source_num_samples} "
        f"({keep_count / source_num_samples:.3%})"
    )


if __name__ == "__main__":
    main()
