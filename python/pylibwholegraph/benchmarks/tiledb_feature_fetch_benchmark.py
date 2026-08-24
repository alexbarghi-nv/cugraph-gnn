# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank feature-gather benchmark for TileDB, pinned CPU, and CUDA.

The benchmark deliberately uses WholeMemory's distributed/NCCL tensor type for
the resident CPU and CUDA baselines.  This matches the communication path used
by the TileDB tensor and makes the storage location the main difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import psutil
import torch  # noqa: TID251 - standalone benchmark, not importable library code

import pylibwholegraph.torch as wgth


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16_777_216)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=[1024, 8192, 65536]
    )
    parser.add_argument(
        "--tile-extents", type=parse_int_list, default=[256, 4096, 65536]
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--backends",
        type=lambda value: [item for item in value.split(",") if item],
        default=["cuda", "cpu", "tiledb"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_chunk(row_start: int, row_end: int, width: int) -> np.ndarray:
    rows = np.arange(row_start, row_end, dtype=np.uint32)[:, None]
    columns = np.arange(width, dtype=np.uint32)[None, :]
    # Exactly representable values with enough variation to catch row/order mistakes.
    return (
        (rows * np.uint32(131) + columns * np.uint32(17)) & np.uint32(0xFFFF)
    ).astype(np.float32)


def prepare_raw_file(path: Path, rows: int, width: int, overwrite: bool) -> None:
    expected_bytes = rows * width * np.dtype(np.float32).itemsize
    if path.is_file() and path.stat().st_size == expected_bytes and not overwrite:
        return
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"{path} exists with the wrong size; pass --overwrite")
        path.unlink()
    chunk_rows = max(1, (64 * 1024**2) // (width * np.dtype(np.float32).itemsize))
    with path.open("wb") as output:
        for row_start in range(0, rows, chunk_rows):
            feature_chunk(row_start, min(rows, row_start + chunk_rows), width).tofile(
                output
            )
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"generated {path} has an unexpected size")


def prepare_tiledb_array(
    raw_path: Path,
    array_path: Path,
    rows: int,
    width: int,
    tile_extent: int,
    overwrite: bool,
) -> None:
    marker = array_path / ".wholememory_benchmark.json"
    expected = {"rows": rows, "width": width, "tile_extent": tile_extent}
    if (
        marker.is_file()
        and json.loads(marker.read_text()) == expected
        and not overwrite
    ):
        return
    if array_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"{array_path} exists with different metadata; pass --overwrite"
            )
        shutil.rmtree(array_path)
    subprocess.run(
        [
            "wholememory_tiledb_ingest",
            str(array_path),
            str(raw_path),
            str(rows),
            str(width * np.dtype(np.float32).itemsize),
            str(tile_extent),
            str(min(rows, 1_048_576)),
        ],
        check=True,
    )
    marker.write_text(json.dumps(expected, sort_keys=True) + "\n")


def drop_file_cache(root: Path) -> None:
    """Ask Linux to evict clean pages for files below root.

    POSIX_FADV_DONTNEED is advisory.  It avoids requiring root or globally
    dropping the machine's page cache, and its effectiveness is reflected in
    the measured ``read_bytes`` counter.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("rb", buffering=0) as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def warm_file_cache(root: Path) -> None:
    """Read every TileDB file once to establish a fully resident cache baseline."""
    buffer = bytearray(8 * 1024**2)
    view = memoryview(buffer)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("rb", buffering=0) as stream:
            while stream.readinto(view):
                pass


def make_traces(
    rows: int,
    batch_sizes: list[int],
    trace_count: int,
    seed: int,
) -> dict[tuple[str, int], list[torch.Tensor]]:
    traces: dict[tuple[str, int], list[torch.Tensor]] = {}
    for batch_size in batch_sizes:
        for pattern_index, pattern in enumerate(("random", "locality")):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + batch_size * 17 + pattern_index)
            host_traces = []
            for _ in range(trace_count):
                if pattern == "random":
                    ids = torch.randint(0, rows, (batch_size,), generator=generator)
                else:
                    window_count = min(16, batch_size)
                    window_rows = min(rows, max(256, batch_size // window_count))
                    starts = torch.randint(
                        0,
                        max(1, rows - window_rows + 1),
                        (window_count,),
                        generator=generator,
                    )
                    windows = torch.randint(
                        0, window_count, (batch_size,), generator=generator
                    )
                    offsets = torch.randint(
                        0, window_rows, (batch_size,), generator=generator
                    )
                    ids = starts[windows] + offsets
                host_traces.append(ids.to(dtype=torch.int64))
            traces[(pattern, batch_size)] = [ids.cuda() for ids in host_traces]
    return traces


def proc_io_bytes(process: psutil.Process) -> int:
    counters = process.io_counters()
    return int(getattr(counters, "read_bytes", 0))


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def benchmark_case(
    tensor: Any,
    traces: list[torch.Tensor],
    warmup: int,
    repetitions: int,
    row_bytes: int,
    cache_mode: str,
    cache_root: Path | None,
) -> dict[str, float]:
    for index in range(warmup):
        output = tensor.gather(traces[index % len(traces)])
        torch.cuda.synchronize()
        del output

    process = psutil.Process()
    latencies_ms: list[float] = []
    read_bytes = 0
    cpu_seconds = 0.0
    for index in range(repetitions):
        if cache_mode == "cold" and cache_root is not None:
            drop_file_cache(cache_root)
        trace = traces[(warmup + index) % len(traces)]
        io_before = proc_io_bytes(process)
        cpu_before = time.process_time_ns()
        start = time.perf_counter()
        output = tensor.gather(trace)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        cpu_after = time.process_time_ns()
        io_after = proc_io_bytes(process)
        # Touch the output after synchronization so an optimizer cannot elide work.
        if not math.isfinite(float(output[0, 0].item())):
            raise RuntimeError("gather returned a non-finite value")
        del output
        latencies_ms.append(elapsed * 1000.0)
        read_bytes += max(0, io_after - io_before)
        cpu_seconds += (cpu_after - cpu_before) / 1_000_000_000.0

    wall_seconds = sum(latencies_ms) / 1000.0
    total_rows = repetitions * traces[0].numel()
    useful_bytes = total_rows * row_bytes
    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "rows_per_second": total_rows / wall_seconds,
        "useful_gib_per_second": useful_bytes / wall_seconds / 1024**3,
        "storage_read_gib": read_bytes / 1024**3,
        "storage_read_gib_per_second": read_bytes / wall_seconds / 1024**3,
        "read_amplification": read_bytes / useful_bytes if useful_bytes else 0.0,
        "process_cpu_percent": cpu_seconds / wall_seconds * 100.0,
        "samples": repetitions,
    }


def verify_tensor(tensor: Any, rows: int, width: int) -> None:
    host_ids = np.asarray([rows - 1, 0, rows // 2, rows - 1], dtype=np.uint32)
    ids = torch.from_numpy(host_ids.astype(np.int64)).cuda()
    actual = tensor.gather(ids).cpu()
    columns = np.arange(width, dtype=np.uint32)[None, :]
    expected_values = (
        (host_ids[:, None] * np.uint32(131) + columns * np.uint32(17))
        & np.uint32(0xFFFF)
    ).astype(np.float32)
    expected = torch.from_numpy(expected_values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def system_metadata(args: argparse.Namespace, raw_path: Path) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    block = subprocess.run(
        ["findmnt", "-T", str(args.data_dir), "-no", "SOURCE,FSTYPE"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": git_commit,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "data_mount": block,
        "rows": args.rows,
        "width": args.width,
        "dtype": "float32",
        "dataset_gib": raw_path.stat().st_size / 1024**3,
        "batch_sizes": args.batch_sizes,
        "tile_extents": args.tile_extents,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "cpu_baseline": "WholeMemory distributed/cpu (cudaMallocHost)",
        "cuda_baseline": "WholeMemory distributed/cuda",
        "cold_cache_method": "POSIX_FADV_DONTNEED per TileDB file before each sample",
    }


def write_results(
    output: Path, metadata: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.width <= 0 or args.repetitions <= 0 or args.warmup < 0:
        raise ValueError(
            "rows, width, and repetitions must be positive; warmup must be nonnegative"
        )
    unknown = set(args.backends) - {"cuda", "cpu", "tiledb"}
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.data_dir / f"features-{args.rows}x{args.width}-float32.bin"
    prepare_raw_file(raw_path, args.rows, args.width, args.overwrite)
    arrays: dict[int, Path] = {}
    if "tiledb" in args.backends:
        for tile_extent in args.tile_extents:
            array_path = args.data_dir / f"features-tile-{tile_extent}.tdb"
            prepare_tiledb_array(
                raw_path, array_path, args.rows, args.width, tile_extent, args.overwrite
            )
            arrays[tile_extent] = array_path
    os.sync()

    trace_count = args.warmup + args.repetitions
    traces = make_traces(args.rows, args.batch_sizes, trace_count, args.seed)
    metadata = system_metadata(args, raw_path)
    results: list[dict[str, Any]] = []
    comm, _ = wgth.init_torch_env_and_create_wm_comm(0, 1, 0, 1, wm_log_level="error")
    try:
        configurations: list[tuple[str, int | None, Path | None]] = []
        if "cuda" in args.backends:
            configurations.append(("cuda", None, None))
        if "cpu" in args.backends:
            configurations.append(("cpu", None, None))
        if "tiledb" in args.backends:
            configurations.extend(
                ("tiledb", extent, arrays[extent]) for extent in args.tile_extents
            )

        for backend, tile_extent, array_path in configurations:
            if backend == "tiledb":
                tensor = wgth.create_wholememory_tensor_from_tiledb(
                    comm,
                    str(array_path),
                    [args.rows, args.width],
                    torch.float32,
                    tensor_entry_partition=[args.rows],
                )
                cache_modes = ("cold", "warm")
            else:
                tensor = wgth.create_wholememory_tensor_from_filelist(
                    comm,
                    "distributed",
                    backend,
                    str(raw_path),
                    torch.float32,
                    last_dim_size=args.width,
                    tensor_entry_partition=np.asarray([args.rows], dtype=np.uintp),
                )
                cache_modes = ("resident",)
            try:
                verify_tensor(tensor, args.rows, args.width)
                for cache_mode in cache_modes:
                    if cache_mode == "warm" and array_path is not None:
                        warm_file_cache(array_path)
                    for pattern in ("random", "locality"):
                        for batch_size in args.batch_sizes:
                            metrics = benchmark_case(
                                tensor,
                                traces[(pattern, batch_size)],
                                args.warmup,
                                args.repetitions,
                                args.width * np.dtype(np.float32).itemsize,
                                cache_mode,
                                array_path,
                            )
                            row = {
                                "backend": backend,
                                "tile_extent_rows": tile_extent,
                                "cache_mode": cache_mode,
                                "pattern": pattern,
                                "batch_size": batch_size,
                                **metrics,
                            }
                            results.append(row)
                            print(json.dumps(row, sort_keys=True), flush=True)
                            write_results(args.output, metadata, results)
            finally:
                wgth.destroy_wholememory_tensor(tensor)
    finally:
        wgth.finalize()

    write_results(args.output, metadata, results)


if __name__ == "__main__":
    main()
