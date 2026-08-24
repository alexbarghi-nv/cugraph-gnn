# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank feature-gather benchmark for TileDB, pinned CPU, and CUDA.

Each process owns one GPU and one rank-local TileDB array. Synthetic IDs are
global, so the normal distributed WholeMemory routing and NCCL path is included.
Results retain every synchronized sample and aggregate throughput using the
slowest rank's latency for each sample.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
from functools import partial
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import psutil
import torch  # noqa: TID251 - standalone benchmark, not importable library code

import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch as wgth
from pylibwholegraph.utils.multiprocess import multiprocess_run


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=16_777_216)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=[1024, 8192, 65536]
    )
    parser.add_argument(
        "--tile-extents", type=parse_int_list, default=[256, 4096, 65536]
    )
    parser.add_argument(
        "--query-chunk-rows",
        type=parse_int_list,
        default=[0],
        help="TileDB unique-row query chunk sizes; zero means one unbounded query",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Optional .npy sampler IDs shaped [samples, ids] or [ranks, samples, ids]",
    )
    parser.add_argument(
        "--backends",
        type=lambda value: [item for item in value.split(",") if item],
        default=["cuda", "cpu", "tiledb"],
    )
    parser.add_argument(
        "--block-device",
        type=Path,
        help="Block device whose sysfs sector counter measures aggregate reads",
    )
    parser.add_argument(
        "--tiledb-stats",
        action="store_true",
        help="Capture TileDB statistics for the first measured sample of each case",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate and vacuum rank-local arrays after ingest",
    )
    parser.add_argument(
        "--storage-baseline",
        action="store_true",
        help="Measure cold sequential reads before the GPU tests",
    )
    parser.add_argument("--storage-baseline-repetitions", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def equal_partition(rows: int, world_size: int) -> tuple[list[int], list[int]]:
    counts = [rows // world_size] * world_size
    for rank in range(rows % world_size):
        counts[rank] += 1
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)
    return counts, offsets


def feature_chunk(row_start: int, row_end: int, width: int) -> np.ndarray:
    rows = np.arange(row_start, row_end, dtype=np.uint32)[:, None]
    columns = np.arange(width, dtype=np.uint32)[None, :]
    return (
        (rows * np.uint32(131) + columns * np.uint32(17)) & np.uint32(0xFFFF)
    ).astype(np.float32)


def prepare_raw_partition(
    path: Path, row_start: int, row_count: int, width: int, overwrite: bool
) -> None:
    expected_bytes = row_count * width * np.dtype(np.float32).itemsize
    if path.is_file() and path.stat().st_size == expected_bytes and not overwrite:
        return
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"{path} exists with the wrong size; pass --overwrite")
        path.unlink()
    chunk_rows = max(1, (64 * 1024**2) // (width * np.dtype(np.float32).itemsize))
    with path.open("wb") as output:
        for local_start in range(0, row_count, chunk_rows):
            global_start = row_start + local_start
            global_end = global_start + min(chunk_rows, row_count - local_start)
            feature_chunk(global_start, global_end, width).tofile(output)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"generated {path} has an unexpected size")


def prepare_tiledb_array(
    raw_path: Path,
    array_path: Path,
    global_row_start: int,
    rows: int,
    width: int,
    tile_extent: int,
    consolidate: bool,
    overwrite: bool,
) -> None:
    marker = array_path / ".wholememory_benchmark.json"
    expected = {
        "global_row_start": global_row_start,
        "rows": rows,
        "width": width,
        "tile_extent": tile_extent,
        "consolidated": consolidate,
    }
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
            "1" if consolidate else "0",
        ],
        check=True,
    )
    marker.write_text(json.dumps(expected, sort_keys=True) + "\n")


def drop_file_cache(root: Path) -> None:
    """Ask Linux to evict clean pages for files below ``root``."""
    if not hasattr(os, "posix_fadvise"):
        return
    paths = root.rglob("*") if root.is_dir() else (root,)
    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb", buffering=0) as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def warm_file_cache(root: Path) -> None:
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
    rank: int,
    trace_file: Path | None,
) -> dict[tuple[str, int], list[torch.Tensor]]:
    traces: dict[tuple[str, int], list[torch.Tensor]] = {}
    for batch_size in batch_sizes:
        for pattern_index, pattern in enumerate(("random", "locality")):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                seed + rank * 1_000_003 + batch_size * 17 + pattern_index
            )
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
    if trace_file is not None:
        recorded = np.load(trace_file, mmap_mode="r")
        if recorded.ndim == 2:
            rank_recorded = recorded
        elif recorded.ndim == 3 and rank < recorded.shape[0]:
            rank_recorded = recorded[rank]
        else:
            raise ValueError(
                "trace file must have shape [samples, ids] or [ranks, samples, ids]"
            )
        if rank_recorded.shape[0] < trace_count:
            raise ValueError("trace file does not contain enough samples")
        for batch_size in batch_sizes:
            if rank_recorded.shape[1] < batch_size:
                raise ValueError(
                    f"trace file does not contain {batch_size} IDs per sample"
                )
            host_traces = []
            for index in range(trace_count):
                ids = np.asarray(rank_recorded[index, :batch_size], dtype=np.int64)
                if np.any(ids < 0) or np.any(ids >= rows):
                    raise ValueError(
                        "trace file contains an ID outside the global row range"
                    )
                host_traces.append(torch.from_numpy(ids.copy()))
            traces[("recorded", batch_size)] = [ids.cuda() for ids in host_traces]
    return traces


def trace_metrics(ids: torch.Tensor, tile_extent: int | None) -> dict[str, int | None]:
    host = np.sort(np.unique(ids.cpu().numpy()))
    ranges = 0 if host.size == 0 else 1 + int(np.count_nonzero(np.diff(host) != 1))
    tiles = (
        int(np.unique(host // tile_extent).size) if tile_extent is not None else None
    )
    return {
        "requested_rows": int(ids.numel()),
        "unique_rows": int(host.size),
        "contiguous_ranges": ranges,
        "estimated_tiles_touched": tiles,
    }


def proc_io_bytes(process: psutil.Process) -> int:
    return int(getattr(process.io_counters(), "read_bytes", 0))


def resolve_block_stat(
    data_dir: Path, override: Path | None
) -> tuple[str, Path | None]:
    source = (
        str(override)
        if override is not None
        else subprocess.run(
            ["findmnt", "-T", str(data_dir), "-no", "SOURCE"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    stat_path = Path("/sys/class/block") / Path(source).name / "stat"
    return source, stat_path if stat_path.is_file() else None


def device_read_bytes(stat_path: Path | None) -> int | None:
    if stat_path is None:
        return None
    return int(stat_path.read_text().split()[2]) * 512


class TileDBStats:
    def __init__(self, requested: bool):
        self.lib = None
        self.error = None
        if not requested:
            return
        for candidate in (None, ctypes.util.find_library("tiledb"), "libtiledb.so"):
            try:
                lib = ctypes.CDLL(candidate) if candidate else ctypes.CDLL(None)
                getattr(lib, "tiledb_stats_enable")
                self.lib = lib
                break
            except (AttributeError, OSError) as error:
                self.error = str(error)
        if self.lib is None:
            return
        self.lib.tiledb_stats_enable.restype = ctypes.c_int32
        self.lib.tiledb_stats_reset.restype = ctypes.c_int32
        self.lib.tiledb_stats_raw_dump_str.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        self.lib.tiledb_stats_raw_dump_str.restype = ctypes.c_int32
        self.lib.tiledb_stats_free_str.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        self.lib.tiledb_stats_free_str.restype = ctypes.c_int32
        if self.lib.tiledb_stats_enable() != 0:
            self.error = "tiledb_stats_enable failed"
            self.lib = None

    @property
    def available(self) -> bool:
        return self.lib is not None

    def reset(self) -> None:
        if self.lib is not None and self.lib.tiledb_stats_reset() != 0:
            raise RuntimeError("tiledb_stats_reset failed")

    def dump(self) -> Any:
        if self.lib is None:
            return None
        output = ctypes.c_char_p()
        if self.lib.tiledb_stats_raw_dump_str(ctypes.byref(output)) != 0:
            raise RuntimeError("tiledb_stats_raw_dump_str failed")
        try:
            raw = ctypes.string_at(output).decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        finally:
            self.lib.tiledb_stats_free_str(ctypes.byref(output))


def verify_tensor(tensor: Any, rows: int, width: int, offsets: list[int]) -> None:
    candidates = [rows - 1, 0, rows // 2]
    candidates.extend(offset for offset in offsets[1:] if offset < rows)
    host_ids = np.asarray(candidates, dtype=np.uint32)
    ids = torch.from_numpy(host_ids.astype(np.int64)).cuda()
    actual = tensor.gather(ids).cpu()
    columns = np.arange(width, dtype=np.uint32)[None, :]
    expected = (
        (host_ids[:, None] * np.uint32(131) + columns * np.uint32(17))
        & np.uint32(0xFFFF)
    ).astype(np.float32)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)


def benchmark_case(
    tensor: Any,
    traces: list[torch.Tensor],
    warmup: int,
    repetitions: int,
    cache_mode: str,
    cache_root: Path | None,
    tile_extent: int | None,
    comm: Any,
    rank: int,
    block_stat: Path | None,
    tiledb_stats: TileDBStats,
) -> list[dict[str, Any]]:
    for index in range(warmup):
        output = tensor.gather(traces[index % len(traces)])
        torch.cuda.synchronize()
        del output
    comm.barrier()

    process = psutil.Process()
    samples: list[dict[str, Any]] = []
    for index in range(repetitions):
        if cache_mode == "cold" and cache_root is not None:
            drop_file_cache(cache_root)
        comm.barrier()
        device_before = device_read_bytes(block_stat) if rank == 0 else None
        if index == 0 and tiledb_stats.available:
            tiledb_stats.reset()
        comm.barrier()

        trace = traces[(warmup + index) % len(traces)]
        io_before = proc_io_bytes(process)
        cpu_before = time.process_time_ns()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = tensor.gather(trace)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        phase_metrics = wmb.get_last_tiledb_gather_metrics()
        cpu_after = time.process_time_ns()
        io_after = proc_io_bytes(process)
        cuda_peak = max(0, torch.cuda.max_memory_allocated() - allocated_before)
        if not math.isfinite(float(output[0, 0].item())):
            raise RuntimeError("gather returned a non-finite value")
        del output
        comm.barrier()
        device_after = device_read_bytes(block_stat) if rank == 0 else None

        sample = {
            "sample": index,
            "rank": rank,
            "latency_ms": elapsed * 1000.0,
            "process_read_bytes": max(0, io_after - io_before),
            "device_read_bytes": (
                max(0, device_after - device_before)
                if device_after is not None and device_before is not None
                else None
            ),
            "cpu_seconds": (cpu_after - cpu_before) / 1_000_000_000.0,
            "cuda_peak_temporary_bytes": cuda_peak,
            "rss_bytes_after": process.memory_info().rss,
            "phase_metrics": phase_metrics,
            **trace_metrics(trace, tile_extent),
        }
        if index == 0 and tiledb_stats.available:
            sample["tiledb_stats"] = tiledb_stats.dump()
        samples.append(sample)
    return samples


def rank_output_path(output: Path, rank: int) -> Path:
    return output.with_name(f"{output.stem}.rank-{rank}{output.suffix}")


def write_rank_results(
    output: Path, rank: int, metadata: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    path = rank_output_path(output, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata, "rank": rank, "results": results}, indent=2)
        + "\n"
    )


def run_rank(
    rank: int,
    world_size: int,
    *,
    args: argparse.Namespace,
    raw_paths: list[Path],
    array_templates: dict[int, str],
    partition: list[int],
    offsets: list[int],
    block_stat: Path | None,
) -> None:
    comm, _ = wgth.init_torch_env_and_create_wm_comm(
        rank, world_size, rank, world_size, wm_log_level="error"
    )
    traces = make_traces(
        args.rows,
        args.batch_sizes,
        args.warmup + args.repetitions,
        args.seed,
        rank,
        args.trace_file,
    )
    results: list[dict[str, Any]] = []
    tiledb_stats = TileDBStats(args.tiledb_stats)
    metadata = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(rank),
        "tiledb_stats_available": tiledb_stats.available,
        "tiledb_stats_error": tiledb_stats.error,
    }
    try:
        configurations: list[tuple[str, int | None, int, Path | None]] = []
        if "cuda" in args.backends:
            configurations.append(("cuda", None, 0, None))
        if "cpu" in args.backends:
            configurations.append(("cpu", None, 0, None))
        if "tiledb" in args.backends:
            for extent in args.tile_extents:
                for query_chunk_rows in args.query_chunk_rows:
                    local_path = Path(
                        array_templates[extent].replace("{rank}", str(rank))
                    )
                    configurations.append(
                        ("tiledb", extent, query_chunk_rows, local_path)
                    )

        for backend, tile_extent, query_chunk_rows, array_path in configurations:
            if backend == "tiledb":
                if query_chunk_rows == 0:
                    os.environ.pop("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS", None)
                else:
                    os.environ["WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS"] = str(
                        query_chunk_rows
                    )
                tensor = wgth.create_wholememory_tensor_from_tiledb(
                    comm,
                    array_templates[tile_extent],
                    [args.rows, args.width],
                    torch.float32,
                    tensor_entry_partition=partition,
                )
                cache_modes = ("cold", "warm")
            else:
                tensor = wgth.create_wholememory_tensor_from_filelist(
                    comm,
                    "distributed",
                    backend,
                    [str(path) for path in raw_paths],
                    torch.float32,
                    last_dim_size=args.width,
                    tensor_entry_partition=np.asarray(partition, dtype=np.uintp),
                )
                cache_modes = ("resident",)
            try:
                verify_tensor(tensor, args.rows, args.width, offsets)
                for cache_mode in cache_modes:
                    if cache_mode == "warm" and array_path is not None:
                        warm_file_cache(array_path)
                    comm.barrier()
                    patterns = ["random", "locality"]
                    if args.trace_file is not None:
                        patterns.append("recorded")
                    for pattern in patterns:
                        for batch_size in args.batch_sizes:
                            samples = benchmark_case(
                                tensor,
                                traces[(pattern, batch_size)],
                                args.warmup,
                                args.repetitions,
                                cache_mode,
                                array_path,
                                tile_extent,
                                comm,
                                rank,
                                block_stat,
                                tiledb_stats,
                            )
                            results.append(
                                {
                                    "backend": backend,
                                    "tile_extent_rows": tile_extent,
                                    "query_chunk_rows": query_chunk_rows,
                                    "cache_mode": cache_mode,
                                    "pattern": pattern,
                                    "batch_size": batch_size,
                                    "samples": samples,
                                }
                            )
                            write_rank_results(args.output, rank, metadata, results)
            finally:
                wgth.destroy_wholememory_tensor(tensor)
        os.environ.pop("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS", None)
    finally:
        wgth.finalize()
    write_rank_results(args.output, rank, metadata, results)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["backend"],
        row["tile_extent_rows"],
        row["query_chunk_rows"],
        row["cache_mode"],
        row["pattern"],
        row["batch_size"],
    )


def aggregate_rank_results(
    output: Path, world_size: int, row_bytes: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shards = [
        json.loads(rank_output_path(output, rank).read_text())
        for rank in range(world_size)
    ]
    rank_metadata = [shard["metadata"] for shard in shards]
    rows_by_rank = [
        {result_key(row): row for row in shard["results"]} for shard in shards
    ]
    keys = list(rows_by_rank[0])
    if any(set(rows) != set(keys) for rows in rows_by_rank[1:]):
        raise RuntimeError("rank result configurations do not match")

    results: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    for key in keys:
        rank_rows = [rows[key] for rows in rows_by_rank]
        sample_count = len(rank_rows[0]["samples"])
        if any(len(row["samples"]) != sample_count for row in rank_rows):
            raise RuntimeError(f"sample count mismatch for {key}")
        aggregate_samples = []
        for sample_index in range(sample_count):
            rank_samples = [row["samples"][sample_index] for row in rank_rows]
            slowest_rank_sample = max(
                rank_samples, key=lambda sample: sample["latency_ms"]
            )
            device_reads = [
                sample["device_read_bytes"]
                for sample in rank_samples
                if sample["device_read_bytes"] is not None
            ]
            process_read_bytes = sum(
                sample["process_read_bytes"] for sample in rank_samples
            )
            requested_rows = sum(sample["requested_rows"] for sample in rank_samples)
            sample = {
                "sample": sample_index,
                "latency_ms": max(sample["latency_ms"] for sample in rank_samples),
                "rank_latencies_ms": [sample["latency_ms"] for sample in rank_samples],
                "process_read_bytes": process_read_bytes,
                "device_read_bytes": max(device_reads) if device_reads else None,
                "measured_read_bytes": max(device_reads)
                if device_reads
                else process_read_bytes,
                "cpu_seconds": sum(sample["cpu_seconds"] for sample in rank_samples),
                "requested_rows": requested_rows,
                "unique_rows": sum(sample["unique_rows"] for sample in rank_samples),
                "contiguous_ranges": sum(
                    sample["contiguous_ranges"] for sample in rank_samples
                ),
                "estimated_tiles_touched": (
                    sum(sample["estimated_tiles_touched"] for sample in rank_samples)
                    if rank_samples[0]["estimated_tiles_touched"] is not None
                    else None
                ),
                "cuda_peak_temporary_bytes": max(
                    sample["cuda_peak_temporary_bytes"] for sample in rank_samples
                ),
                "rss_bytes_after": sum(
                    sample["rss_bytes_after"] for sample in rank_samples
                ),
                "rank_tiledb_stats": [
                    sample.get("tiledb_stats") for sample in rank_samples
                ],
                "staging_allocation_ms": slowest_rank_sample["phase_metrics"][
                    "staging_allocation_ms"
                ],
                "indices_d2h_ms": slowest_rank_sample["phase_metrics"][
                    "indices_d2h_ms"
                ],
                "tiledb_read_ms": slowest_rank_sample["phase_metrics"][
                    "tiledb_read_ms"
                ],
                "rows_h2d_ms": slowest_rank_sample["phase_metrics"]["rows_h2d_ms"],
                "index_bytes": slowest_rank_sample["phase_metrics"]["index_bytes"],
                "raw_staging_bytes": slowest_rank_sample["phase_metrics"][
                    "raw_staging_bytes"
                ],
                "output_bytes": slowest_rank_sample["phase_metrics"]["output_bytes"],
            }
            aggregate_samples.append(sample)
            raw_samples.append(
                {
                    "backend": key[0],
                    "tile_extent_rows": key[1],
                    "query_chunk_rows": key[2],
                    "cache_mode": key[3],
                    "pattern": key[4],
                    "batch_size": key[5],
                    **sample,
                }
            )

        latencies = [sample["latency_ms"] for sample in aggregate_samples]
        wall_seconds = sum(latencies) / 1000.0
        total_rows = sum(sample["requested_rows"] for sample in aggregate_samples)
        useful_bytes = total_rows * row_bytes
        storage_read_bytes = sum(
            sample["measured_read_bytes"] for sample in aggregate_samples
        )
        process_read_bytes = sum(
            sample["process_read_bytes"] for sample in aggregate_samples
        )
        device_values = [
            sample["device_read_bytes"]
            for sample in aggregate_samples
            if sample["device_read_bytes"] is not None
        ]
        results.append(
            {
                "backend": key[0],
                "tile_extent_rows": key[1],
                "query_chunk_rows": key[2],
                "cache_mode": key[3],
                "pattern": key[4],
                "batch_size": key[5],
                "world_size": world_size,
                "latency_mean_ms": statistics.mean(latencies),
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "rows_per_second": total_rows / wall_seconds,
                "useful_gib_per_second": useful_bytes / wall_seconds / 1024**3,
                "storage_read_gib": storage_read_bytes / 1024**3,
                "storage_read_gib_per_second": storage_read_bytes
                / wall_seconds
                / 1024**3,
                "process_read_gib": process_read_bytes / 1024**3,
                "device_read_gib": (
                    sum(device_values) / 1024**3 if device_values else None
                ),
                "storage_counter_source": "device" if device_values else "process",
                "read_amplification": storage_read_bytes / useful_bytes
                if useful_bytes
                else 0.0,
                "process_cpu_percent": sum(
                    sample["cpu_seconds"] for sample in aggregate_samples
                )
                / wall_seconds
                * 100.0,
                "unique_row_fraction": sum(
                    sample["unique_rows"] for sample in aggregate_samples
                )
                / total_rows,
                "contiguous_ranges_mean": statistics.mean(
                    sample["contiguous_ranges"] for sample in aggregate_samples
                ),
                "estimated_tiles_touched_mean": (
                    statistics.mean(
                        sample["estimated_tiles_touched"]
                        for sample in aggregate_samples
                    )
                    if aggregate_samples[0]["estimated_tiles_touched"] is not None
                    else None
                ),
                "cuda_peak_temporary_bytes": max(
                    sample["cuda_peak_temporary_bytes"] for sample in aggregate_samples
                ),
                "aggregate_rss_peak_bytes": max(
                    sample["rss_bytes_after"] for sample in aggregate_samples
                ),
                "staging_allocation_mean_ms": statistics.mean(
                    sample["staging_allocation_ms"] for sample in aggregate_samples
                ),
                "indices_d2h_mean_ms": statistics.mean(
                    sample["indices_d2h_ms"] for sample in aggregate_samples
                ),
                "tiledb_read_mean_ms": statistics.mean(
                    sample["tiledb_read_ms"] for sample in aggregate_samples
                ),
                "rows_h2d_mean_ms": statistics.mean(
                    sample["rows_h2d_ms"] for sample in aggregate_samples
                ),
                "index_bytes_peak_per_rank": max(
                    sample["index_bytes"] for sample in aggregate_samples
                ),
                "raw_staging_bytes_peak_per_rank": max(
                    sample["raw_staging_bytes"] for sample in aggregate_samples
                ),
                "output_bytes_peak_per_rank": max(
                    sample["output_bytes"] for sample in aggregate_samples
                ),
                "samples": sample_count,
            }
        )
    return results, raw_samples, rank_metadata


def benchmark_storage_baseline(
    raw_paths: list[Path], repetitions: int, block_stat: Path | None
) -> dict[str, Any]:
    process = psutil.Process()
    samples = []
    total_bytes = sum(path.stat().st_size for path in raw_paths)
    buffer = bytearray(8 * 1024**2)
    view = memoryview(buffer)
    for sample_index in range(repetitions):
        for path in raw_paths:
            drop_file_cache(path)
        device_before = device_read_bytes(block_stat)
        process_before = proc_io_bytes(process)
        start = time.perf_counter()
        for path in raw_paths:
            with path.open("rb", buffering=0) as stream:
                while stream.readinto(view):
                    pass
        elapsed = time.perf_counter() - start
        process_after = proc_io_bytes(process)
        device_after = device_read_bytes(block_stat)
        measured_bytes = (
            max(0, device_after - device_before)
            if device_after is not None and device_before is not None
            else max(0, process_after - process_before)
        )
        samples.append(
            {
                "sample": sample_index,
                "latency_seconds": elapsed,
                "logical_gib_per_second": total_bytes / elapsed / 1024**3,
                "measured_read_gib_per_second": measured_bytes / elapsed / 1024**3,
                "measured_read_bytes": measured_bytes,
            }
        )
    return {
        "method": "cold buffered sequential read of all rank partitions",
        "total_bytes": total_bytes,
        "throughput_mean_gib_per_second": statistics.mean(
            sample["measured_read_gib_per_second"] for sample in samples
        ),
        "samples": samples,
    }


def system_metadata(
    args: argparse.Namespace,
    raw_paths: list[Path],
    block_device: str,
    block_stat: Path | None,
) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_commit": git_commit,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(rank) for rank in range(args.world_size)],
        "world_size": args.world_size,
        "data_mount": subprocess.run(
            ["findmnt", "-T", str(args.data_dir), "-no", "SOURCE,FSTYPE"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "block_device": block_device,
        "block_stat": str(block_stat) if block_stat is not None else None,
        "rows": args.rows,
        "width": args.width,
        "dtype": "float32",
        "dataset_gib": sum(path.stat().st_size for path in raw_paths) / 1024**3,
        "batch_sizes": args.batch_sizes,
        "tile_extents": args.tile_extents,
        "query_chunk_rows": args.query_chunk_rows,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "patterns": [
            "random",
            "locality",
            *(["recorded"] if args.trace_file is not None else []),
        ],
        "trace_file": str(args.trace_file) if args.trace_file is not None else None,
        "consolidated": args.consolidate,
        "cpu_baseline": "WholeMemory distributed/cpu (cudaMallocHost)",
        "cuda_baseline": "WholeMemory distributed/cuda",
        "cold_cache_method": "POSIX_FADV_DONTNEED per rank-local TileDB file before each sample",
        "aggregate_latency": "maximum rank latency per synchronized sample",
        "storage_counter": "block-device sectors when available, otherwise summed process read_bytes",
    }


def write_results(
    output: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    rank_metadata: list[dict[str, Any]],
    storage_baseline: dict[str, Any] | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "rank_metadata": rank_metadata,
                "storage_baseline": storage_baseline,
                "results": results,
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    sample_rows = [
        {key: value for key, value in sample.items() if not isinstance(value, list)}
        for sample in samples
    ]
    with output.with_name(f"{output.stem}.samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)


def main() -> None:
    args = parse_args()
    if (
        args.rows <= 0
        or args.width <= 0
        or args.world_size <= 0
        or args.repetitions <= 0
        or args.warmup < 0
        or args.storage_baseline_repetitions <= 0
    ):
        raise ValueError(
            "sizes and repetitions must be positive; warmup must be nonnegative"
        )
    if args.rows < args.world_size:
        raise ValueError("rows must be at least world-size")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"requested {args.world_size} ranks but found {torch.cuda.device_count()} GPUs"
        )
    unknown = set(args.backends) - {"cuda", "cpu", "tiledb"}
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")
    if any(value < 0 for value in args.query_chunk_rows):
        raise ValueError("query chunk rows must be nonnegative")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    partition, offsets = equal_partition(args.rows, args.world_size)
    raw_paths = []
    for rank, (row_count, row_offset) in enumerate(
        zip(partition, offsets, strict=True)
    ):
        raw_path = args.data_dir / (
            f"features-{args.rows}x{args.width}-float32-part-{rank}-of-{args.world_size}.bin"
        )
        prepare_raw_partition(
            raw_path, row_offset, row_count, args.width, args.overwrite
        )
        raw_paths.append(raw_path)

    array_templates: dict[int, str] = {}
    suffix = "-consolidated" if args.consolidate else ""
    if "tiledb" in args.backends:
        for extent in args.tile_extents:
            template = str(
                args.data_dir
                / f"features-tile-{extent}{suffix}-rank-{{rank}}-of-{args.world_size}.tdb"
            )
            array_templates[extent] = template
            for rank, (row_count, row_offset) in enumerate(
                zip(partition, offsets, strict=True)
            ):
                prepare_tiledb_array(
                    raw_paths[rank],
                    Path(template.replace("{rank}", str(rank))),
                    row_offset,
                    row_count,
                    args.width,
                    extent,
                    args.consolidate,
                    args.overwrite,
                )
    os.sync()

    block_device, block_stat = resolve_block_stat(args.data_dir, args.block_device)
    storage_baseline = (
        benchmark_storage_baseline(
            raw_paths, args.storage_baseline_repetitions, block_stat
        )
        if args.storage_baseline
        else None
    )
    metadata = system_metadata(args, raw_paths, block_device, block_stat)
    worker = partial(
        run_rank,
        args=args,
        raw_paths=raw_paths,
        array_templates=array_templates,
        partition=partition,
        offsets=offsets,
        block_stat=block_stat,
    )
    multiprocess_run(args.world_size, worker, inline_single_process=True)
    results, samples, rank_metadata = aggregate_rank_results(
        args.output, args.world_size, args.width * np.dtype(np.float32).itemsize
    )
    write_results(
        args.output,
        metadata,
        results,
        samples,
        rank_metadata,
        storage_baseline,
    )
    for row in results:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
