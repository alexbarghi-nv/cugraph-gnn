# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Single-node WholeMemory loading benchmark for TileDB and pinned CPU storage.

Each process owns one GPU. TileDB storage may use one array per rank or one
communicator-shared array. Synthetic IDs are global, so normal distributed
WholeMemory routing and the single NCCL communicator are included.
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

PHASE_TIMING_FIELDS = (
    "id_routing_ms",
    "gpu_sort_ms",
    "gpu_deduplicate_ms",
    "staging_allocation_ms",
    "indices_d2h_ms",
    "tiledb_read_ms",
    "id_decode_ms",
    "id_sort_ms",
    "id_deduplicate_ms",
    "query_setup_ms",
    "range_build_ms",
    "query_submit_ms",
    "cpu_reorder_ms",
    "rows_h2d_ms",
    "gpu_expand_ms",
    "embedding_exchange_ms",
    "output_reorder_ms",
)
PHASE_COUNT_FIELDS = (
    "storage_requested_rows",
    "storage_unique_rows",
    "storage_range_count",
    "storage_query_count",
    "index_bytes",
    "raw_staging_bytes",
    "output_bytes",
)
TILEDB_STAT_TIMING_FIELDS = (
    "tile_overlap_planning_ms",
    "relevant_tile_overlap_ms",
    "partition_planning_ms",
    "internal_tile_read_ms",
    "internal_unfilter_ms",
    "internal_copy_ms",
    "internal_reader_work_ms",
)
TILEDB_STAT_COUNT_FIELDS = (
    "tiledb_ranges_requested",
    "tiledb_tiles_read",
    "tiledb_vfs_read_ops",
    "tiledb_vfs_read_bytes",
)

OVERLAP_CASES = {
    "independent": {"within_rank_repeat": 1, "cross_rank_shared": False},
    "cross_rank_25": {"within_rank_repeat": 1, "cross_rank_shared": True},
    "within_rank_25": {"within_rank_repeat": 4, "cross_rank_shared": False},
    "combined_12_5": {"within_rank_repeat": 2, "cross_rank_shared": True},
    "combined_6_25": {"within_rank_repeat": 4, "cross_rank_shared": True},
    "combined_3_125": {"within_rank_repeat": 8, "cross_rank_shared": True},
    "stress_1": {"within_rank_repeat": 25, "cross_rank_shared": True},
}
OVERLAP_PLACEMENTS = ("clustered", "scattered")


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
        "--array-layouts",
        type=lambda value: [item for item in value.split(",") if item],
        default=["rank", "node"],
        help="TileDB physical layouts to test: rank,node",
    )
    parser.add_argument(
        "--locality-window-rows",
        type=parse_int_list,
        default=[256, 4096, 65536],
        help="Generate locality traces restricted to one random window of each size",
    )
    parser.add_argument(
        "--patterns",
        type=lambda value: [item for item in value.split(",") if item],
        help=(
            "Access patterns to run; defaults to random plus every configured "
            "window and the recorded trace when supplied"
        ),
    )
    parser.add_argument(
        "--overlap-cases",
        type=lambda value: [item for item in value.split(",") if item],
        default=[],
        help=(
            "Generate exact synthetic overlap cases: "
            + ",".join(OVERLAP_CASES)
        ),
    )
    parser.add_argument(
        "--overlap-placements",
        type=lambda value: [item for item in value.split(",") if item],
        default=[],
        help="Unique-row placement for overlap cases: clustered,scattered",
    )
    parser.add_argument(
        "--cache-modes",
        type=lambda value: [item for item in value.split(",") if item],
        default=["cold", "warm"],
        help="TileDB cache modes to run: cold,warm",
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
        default=["cpu", "tiledb"],
    )
    parser.add_argument(
        "--block-device",
        type=Path,
        help="Block device whose sysfs sector counter measures aggregate reads",
    )
    parser.add_argument(
        "--tiledb-stats",
        action="store_true",
        help="Capture compact TileDB statistics for every measured TileDB sample",
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
    input_row_offset: int = 0,
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
            str(input_row_offset),
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


def overlap_pattern(case: str, placement: str) -> str:
    return f"overlap_{placement}_{case}"


def overlap_pattern_spec(pattern: str) -> dict[str, Any] | None:
    for placement in OVERLAP_PLACEMENTS:
        prefix = f"overlap_{placement}_"
        if pattern.startswith(prefix):
            case = pattern[len(prefix) :]
            if case in OVERLAP_CASES:
                return {
                    "case": case,
                    "placement": placement,
                    **OVERLAP_CASES[case],
                }
    return None


def coprime_multiplier(rows: int, seed: int) -> int:
    candidate = 1 + 2 * (seed % max(1, rows // 2))
    while math.gcd(candidate, rows) != 1:
        candidate += 2
        if candidate >= rows:
            candidate = 1
    return candidate


def make_overlap_host_ids(
    rows: int,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    case: str,
    placement: str,
    sample_index: int,
) -> np.ndarray:
    spec = OVERLAP_CASES[case]
    repeat = int(spec["within_rank_repeat"])
    shared = bool(spec["cross_rank_shared"])
    if batch_size % repeat != 0:
        raise ValueError(
            f"batch size {batch_size} must be divisible by overlap repeat {repeat}"
        )
    unique_count = batch_size // repeat
    required_unique_rows = unique_count if shared else unique_count * world_size
    if required_unique_rows > rows:
        raise ValueError(
            f"overlap case {case} requires {required_unique_rows} distinct rows, "
            f"but the array contains {rows}"
        )

    case_index = list(OVERLAP_CASES).index(case)
    placement_index = OVERLAP_PLACEMENTS.index(placement)
    trace_seed = (
        seed
        + batch_size * 17
        + sample_index * 104_729
        + case_index * 1_000_003
        + placement_index * 10_000_019
    )
    rank_slot = 0 if shared else rank
    ordinal_start = rank_slot * unique_count
    ordinals = np.arange(
        ordinal_start, ordinal_start + unique_count, dtype=np.int64
    )
    if placement == "clustered":
        span_start = trace_seed % (rows - required_unique_rows + 1)
        unique_ids = ordinals + span_start
    elif placement == "scattered":
        multiplier = coprime_multiplier(rows, trace_seed)
        offset = (trace_seed * 6_364_136_223_846_793_005 + 1) % rows
        unique_ids = (ordinals * multiplier + offset) % rows
    else:
        raise ValueError(f"unknown overlap placement: {placement}")

    ids = np.repeat(unique_ids, repeat)
    generator = np.random.default_rng(trace_seed + rank * 1_000_000_007)
    return generator.permutation(ids).astype(np.int64, copy=False)


def make_overlap_ids(
    rows: int,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    case: str,
    placement: str,
    sample_index: int,
) -> torch.Tensor:
    return torch.from_numpy(
        make_overlap_host_ids(
            rows,
            batch_size,
            seed,
            rank,
            world_size,
            case,
            placement,
            sample_index,
        )
    )


def make_traces(
    rows: int,
    batch_sizes: list[int],
    trace_count: int,
    seed: int,
    rank: int,
    world_size: int,
    trace_file: Path | None,
    locality_window_rows: list[int],
    overlap_cases: list[str],
    overlap_placements: list[str],
    selected_patterns: list[str],
) -> dict[tuple[str, int], list[torch.Tensor]]:
    traces: dict[tuple[str, int], list[torch.Tensor]] = {}
    patterns: list[tuple[str, int | None]] = [("random", None)]
    patterns.extend(
        (f"window_{window_rows}", window_rows) for window_rows in locality_window_rows
    )
    for batch_size in batch_sizes:
        for pattern_index, (pattern, window_rows) in enumerate(patterns):
            if pattern not in selected_patterns:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                seed + rank * 1_000_003 + batch_size * 17 + pattern_index
            )
            host_traces = []
            for _ in range(trace_count):
                if window_rows is None:
                    ids = torch.randint(0, rows, (batch_size,), generator=generator)
                else:
                    effective_window_rows = min(rows, window_rows)
                    start = torch.randint(
                        0,
                        max(1, rows - effective_window_rows + 1),
                        (),
                        generator=generator,
                    )
                    offsets = torch.randint(
                        0, effective_window_rows, (batch_size,), generator=generator
                    )
                    ids = start + offsets
                host_traces.append(ids.to(dtype=torch.int64))
            traces[(pattern, batch_size)] = [ids.cuda() for ids in host_traces]
        for placement in overlap_placements:
            for case in overlap_cases:
                pattern = overlap_pattern(case, placement)
                if pattern not in selected_patterns:
                    continue
                host_traces = [
                    make_overlap_ids(
                        rows,
                        batch_size,
                        seed,
                        rank,
                        world_size,
                        case,
                        placement,
                        sample_index,
                    )
                    for sample_index in range(trace_count)
                ]
                traces[(pattern, batch_size)] = [
                    ids.cuda() for ids in host_traces
                ]
    if trace_file is not None and "recorded" in selected_patterns:
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


def trace_metrics(
    ids: torch.Tensor,
    tile_extent: int | None,
    owner_boundaries: list[int],
) -> dict[str, Any]:
    host = np.sort(np.unique(ids.cpu().numpy()))
    ranges = 0 if host.size == 0 else 1 + int(np.count_nonzero(np.diff(host) != 1))
    tiles = (
        int(np.unique(host // tile_extent).size) if tile_extent is not None else None
    )
    owners = np.searchsorted(
        np.asarray(owner_boundaries[1:], dtype=np.int64), host, side="right"
    )
    owner_unique_counts = np.bincount(
        owners, minlength=len(owner_boundaries) - 1
    ).tolist()
    return {
        "requested_rows": int(ids.numel()),
        "unique_rows": int(host.size),
        "contiguous_ranges": ranges,
        "estimated_tiles_touched": tiles,
        "owner_unique_counts": owner_unique_counts,
    }


def aggregate_trace_overlap_metrics(
    rank_samples: list[dict[str, Any]], pattern: str, world_size: int
) -> dict[str, Any]:
    pattern_spec = overlap_pattern_spec(pattern)
    requested_rows = sum(sample["requested_rows"] for sample in rank_samples)
    within_rank_unique_rows = sum(sample["unique_rows"] for sample in rank_samples)
    if pattern_spec is None:
        node_unique_rows = None
        requesting_ranks_per_unique_row = None
        owner_unique_counts = None
    elif pattern_spec["cross_rank_shared"]:
        node_unique_rows = rank_samples[0]["unique_rows"]
        owner_unique_counts = rank_samples[0]["owner_unique_counts"]
        requesting_ranks_per_unique_row = world_size
    else:
        node_unique_rows = within_rank_unique_rows
        owner_unique_counts = [
            sum(sample["owner_unique_counts"][owner] for sample in rank_samples)
            for owner in range(world_size)
        ]
        requesting_ranks_per_unique_row = 1
    owner_unique_mean = (
        statistics.mean(owner_unique_counts)
        if owner_unique_counts is not None
        else None
    )
    owner_unique_max = (
        max(owner_unique_counts) if owner_unique_counts is not None else None
    )
    return {
        "requested_rows": requested_rows,
        "unique_rows": within_rank_unique_rows,
        "node_unique_rows": node_unique_rows,
        "within_rank_unique_row_fraction": within_rank_unique_rows / requested_rows,
        "node_unique_row_fraction": (
            node_unique_rows / requested_rows
            if node_unique_rows is not None
            else None
        ),
        "within_rank_repetition": requested_rows / within_rank_unique_rows,
        "requesting_ranks_per_unique_row": requesting_ranks_per_unique_row,
        "owner_unique_rows_mean": owner_unique_mean,
        "owner_unique_rows_max": owner_unique_max,
        "owner_unique_max_to_mean": (
            owner_unique_max / owner_unique_mean
            if owner_unique_mean is not None and owner_unique_mean > 0
            else None
        ),
        "owner_unique_counts": owner_unique_counts,
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

    def dump_selected(self) -> dict[str, float | int]:
        raw = self.dump()
        contexts = (
            raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
        )

        def timer(name: str) -> float:
            return 1000.0 * sum(
                float(context.get("timers", {}).get(name, 0.0))
                for context in contexts
                if isinstance(context, dict)
            )

        def counter(name: str) -> int:
            return sum(
                int(context.get("counters", {}).get(name, 0))
                for context in contexts
                if isinstance(context, dict)
            )

        return {
            "tile_overlap_planning_ms": timer(
                "Context.subSubarray.read_compute_tile_overlap.sum"
            ),
            "relevant_tile_overlap_ms": timer(
                "Context.subSubarray.read_compute_relevant_tile_overlap.sum"
            ),
            "partition_planning_ms": timer(
                "Context.Query.Reader.SubarrayPartitioner.read_next_partition.sum"
            ),
            "internal_tile_read_ms": timer("Context.Query.Reader.read_tiles.sum"),
            "internal_unfilter_ms": timer(
                "Context.Query.Reader.unfilter_attr_tiles.sum"
            ),
            "internal_copy_ms": timer("Context.Query.Reader.copy_fixed_tiles.sum"),
            "internal_reader_work_ms": timer("Context.Query.Reader.dowork.sum"),
            "tiledb_ranges_requested": counter(
                "Context.subSubarray.precompute_tile_overlap.ranges_requested"
            ),
            "tiledb_tiles_read": counter("Context.Query.Reader.num_tiles_read"),
            "tiledb_vfs_read_ops": counter("Context.VFS.read_ops_num"),
            "tiledb_vfs_read_bytes": counter("Context.VFS.read_byte_num"),
        }


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
    shared_cache_root: bool,
    owner_boundaries: list[int],
) -> list[dict[str, Any]]:
    for index in range(warmup):
        output = tensor.gather(traces[index % len(traces)])
        torch.cuda.synchronize()
        del output
    comm.barrier()

    process = psutil.Process()
    samples: list[dict[str, Any]] = []
    for index in range(repetitions):
        if (
            cache_mode == "cold"
            and cache_root is not None
            and (not shared_cache_root or rank == 0)
        ):
            drop_file_cache(cache_root)
        comm.barrier()
        device_before = device_read_bytes(block_stat) if rank == 0 else None
        collect_tiledb_stats = cache_root is not None and tiledb_stats.available
        if collect_tiledb_stats:
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
            **trace_metrics(trace, tile_extent, owner_boundaries),
        }
        if collect_tiledb_stats:
            sample["tiledb_stats"] = tiledb_stats.dump_selected()
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
    array_templates: dict[tuple[int, str], str],
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
        world_size,
        args.trace_file,
        args.locality_window_rows,
        args.overlap_cases,
        args.overlap_placements,
        args.patterns,
    )
    results: list[dict[str, Any]] = []
    tiledb_stats = TileDBStats(args.tiledb_stats)
    metadata = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(rank),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "tiledb_stats_available": tiledb_stats.available,
        "tiledb_stats_error": tiledb_stats.error,
    }
    try:
        configurations: list[tuple[str, str, int | None, int, Path | None]] = []
        if "cuda" in args.backends:
            configurations.append(("cuda", "none", None, 0, None))
        if "cpu" in args.backends:
            configurations.append(("cpu", "none", None, 0, None))
        if "tiledb" in args.backends:
            for array_layout in args.array_layouts:
                for extent in args.tile_extents:
                    for query_chunk_rows in args.query_chunk_rows:
                        uri = array_templates[(extent, array_layout)]
                        local_path = Path(uri.replace("{rank}", str(rank)))
                        configurations.append(
                            (
                                "tiledb",
                                array_layout,
                                extent,
                                query_chunk_rows,
                                local_path,
                            )
                        )

        for (
            backend,
            array_layout,
            tile_extent,
            query_chunk_rows,
            array_path,
        ) in configurations:
            if backend == "tiledb":
                if query_chunk_rows == 0:
                    os.environ.pop("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS", None)
                else:
                    os.environ["WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS"] = str(
                        query_chunk_rows
                    )
                tensor = wgth.create_wholememory_tensor_from_tiledb(
                    comm,
                    array_templates[(tile_extent, array_layout)],
                    [args.rows, args.width],
                    torch.float32,
                    tensor_entry_partition=partition,
                    array_layout=array_layout,
                )
                cache_modes = args.cache_modes
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
                        if array_layout == "rank" or rank == 0:
                            warm_file_cache(array_path)
                    comm.barrier()
                    for pattern in args.patterns:
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
                                array_layout == "node",
                                [*offsets, args.rows],
                            )
                            results.append(
                                {
                                    "backend": backend,
                                    "array_layout": array_layout,
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
        row["array_layout"],
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
        pattern_spec = overlap_pattern_spec(key[5])
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
            trace_summary = aggregate_trace_overlap_metrics(
                rank_samples, key[5], world_size
            )
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
                **trace_summary,
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
                "rank_phase_metrics": [
                    sample["phase_metrics"] for sample in rank_samples
                ],
                **{
                    field: slowest_rank_sample["phase_metrics"][field]
                    for field in (*PHASE_TIMING_FIELDS, *PHASE_COUNT_FIELDS)
                },
                **{
                    field: slowest_rank_sample.get("tiledb_stats", {}).get(field)
                    for field in (*TILEDB_STAT_TIMING_FIELDS, *TILEDB_STAT_COUNT_FIELDS)
                },
            }
            aggregate_samples.append(sample)
            raw_samples.append(
                {
                    "backend": key[0],
                    "array_layout": key[1],
                    "tile_extent_rows": key[2],
                    "query_chunk_rows": key[3],
                    "cache_mode": key[4],
                    "pattern": key[5],
                    "batch_size": key[6],
                    "overlap_case": (
                        pattern_spec["case"] if pattern_spec is not None else None
                    ),
                    "overlap_placement": (
                        pattern_spec["placement"]
                        if pattern_spec is not None
                        else None
                    ),
                    "width": row_bytes // np.dtype(np.float32).itemsize,
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
        result = {
            "backend": key[0],
            "array_layout": key[1],
            "tile_extent_rows": key[2],
            "query_chunk_rows": key[3],
            "cache_mode": key[4],
            "pattern": key[5],
            "batch_size": key[6],
            "overlap_case": (
                pattern_spec["case"] if pattern_spec is not None else None
            ),
            "overlap_placement": (
                pattern_spec["placement"] if pattern_spec is not None else None
            ),
            "width": row_bytes // np.dtype(np.float32).itemsize,
            "world_size": world_size,
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "rows_per_second": total_rows / wall_seconds,
            "useful_gib_per_second": useful_bytes / wall_seconds / 1024**3,
            "storage_read_gib": storage_read_bytes / 1024**3,
            "storage_read_gib_per_second": storage_read_bytes / wall_seconds / 1024**3,
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
            "within_rank_unique_row_fraction": sum(
                sample["unique_rows"] for sample in aggregate_samples
            )
            / total_rows,
            "node_unique_row_fraction": (
                sum(sample["node_unique_rows"] for sample in aggregate_samples)
                / total_rows
                if aggregate_samples[0]["node_unique_rows"] is not None
                else None
            ),
            "within_rank_repetition_mean": statistics.mean(
                sample["within_rank_repetition"] for sample in aggregate_samples
            ),
            "requesting_ranks_per_unique_row_mean": (
                statistics.mean(
                    sample["requesting_ranks_per_unique_row"]
                    for sample in aggregate_samples
                )
                if aggregate_samples[0]["requesting_ranks_per_unique_row"]
                is not None
                else None
            ),
            "owner_unique_rows_mean": (
                statistics.mean(
                    sample["owner_unique_rows_mean"]
                    for sample in aggregate_samples
                )
                if aggregate_samples[0]["owner_unique_rows_mean"] is not None
                else None
            ),
            "owner_unique_rows_max_mean": (
                statistics.mean(
                    sample["owner_unique_rows_max"]
                    for sample in aggregate_samples
                )
                if aggregate_samples[0]["owner_unique_rows_max"] is not None
                else None
            ),
            "owner_unique_max_to_mean_mean": (
                statistics.mean(
                    sample["owner_unique_max_to_mean"]
                    for sample in aggregate_samples
                )
                if aggregate_samples[0]["owner_unique_max_to_mean"] is not None
                else None
            ),
            "contiguous_ranges_mean": statistics.mean(
                sample["contiguous_ranges"] for sample in aggregate_samples
            ),
            "estimated_tiles_touched_mean": (
                statistics.mean(
                    sample["estimated_tiles_touched"] for sample in aggregate_samples
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
        for field in PHASE_TIMING_FIELDS:
            result[f"{field[:-3]}_mean_ms"] = statistics.mean(
                sample[field] for sample in aggregate_samples
            )
        for field in PHASE_COUNT_FIELDS[:4]:
            result[f"{field}_mean"] = statistics.mean(
                sample[field] for sample in aggregate_samples
            )
        for field in TILEDB_STAT_TIMING_FIELDS:
            values = [
                sample[field]
                for sample in aggregate_samples
                if sample[field] is not None
            ]
            result[f"{field[:-3]}_mean_ms"] = (
                statistics.mean(values) if values else None
            )
        for field in TILEDB_STAT_COUNT_FIELDS:
            values = [
                sample[field]
                for sample in aggregate_samples
                if sample[field] is not None
            ]
            result[f"{field}_mean"] = statistics.mean(values) if values else None
        results.append(result)
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
        "array_layouts": args.array_layouts,
        "cache_modes": args.cache_modes,
        "locality_window_rows": args.locality_window_rows,
        "query_chunk_rows": args.query_chunk_rows,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "patterns": args.patterns,
        "overlap_cases": args.overlap_cases,
        "overlap_placements": args.overlap_placements,
        "overlap_case_definitions": OVERLAP_CASES,
        "trace_file": str(args.trace_file) if args.trace_file is not None else None,
        "consolidated": args.consolidate,
        "tiledb_compute_concurrency": os.getenv(
            "WHOLEMEMORY_TILEDB_COMPUTE_CONCURRENCY"
        ),
        "tiledb_io_concurrency": os.getenv("WHOLEMEMORY_TILEDB_IO_CONCURRENCY"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_baseline": "WholeMemory distributed/cpu (cudaMallocHost)",
        "cuda_baseline": "WholeMemory distributed/cuda",
        "cold_cache_method": "POSIX_FADV_DONTNEED before each sample; rank 0 evicts node-shared arrays",
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
    unknown_layouts = set(args.array_layouts) - {"rank", "node"}
    if unknown_layouts or not args.array_layouts:
        raise ValueError(
            f"array layouts must contain rank and/or node, got {sorted(unknown_layouts)}"
        )
    if any(window_rows <= 0 for window_rows in args.locality_window_rows):
        raise ValueError("locality window rows must be positive")
    unknown_overlap_cases = set(args.overlap_cases) - set(OVERLAP_CASES)
    if unknown_overlap_cases:
        raise ValueError(f"unknown overlap cases: {sorted(unknown_overlap_cases)}")
    unknown_overlap_placements = set(args.overlap_placements) - set(
        OVERLAP_PLACEMENTS
    )
    if unknown_overlap_placements:
        raise ValueError(
            f"unknown overlap placements: {sorted(unknown_overlap_placements)}"
        )
    if bool(args.overlap_cases) != bool(args.overlap_placements):
        raise ValueError(
            "overlap cases and overlap placements must be configured together"
        )
    for case in args.overlap_cases:
        repeat = int(OVERLAP_CASES[case]["within_rank_repeat"])
        if any(batch_size % repeat != 0 for batch_size in args.batch_sizes):
            raise ValueError(
                f"every batch size must be divisible by {repeat} for overlap "
                f"case {case}"
            )
    unknown_cache_modes = set(args.cache_modes) - {"cold", "warm"}
    if unknown_cache_modes or not args.cache_modes:
        raise ValueError(
            f"cache modes must contain cold and/or warm, got {sorted(unknown_cache_modes)}"
        )
    overlap_patterns = [
        overlap_pattern(case, placement)
        for placement in args.overlap_placements
        for case in args.overlap_cases
    ]
    available_patterns = {
        "random",
        *(f"window_{window_rows}" for window_rows in args.locality_window_rows),
        *overlap_patterns,
        *(["recorded"] if args.trace_file is not None else []),
    }
    if args.patterns is None:
        args.patterns = [
            "random",
            *(f"window_{window_rows}" for window_rows in args.locality_window_rows),
            *overlap_patterns,
            *(["recorded"] if args.trace_file is not None else []),
        ]
    unknown_patterns = set(args.patterns) - available_patterns
    if unknown_patterns or not args.patterns:
        raise ValueError(
            f"patterns are not generated by this invocation: {sorted(unknown_patterns)}"
        )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    partition, offsets = equal_partition(args.rows, args.world_size)
    raw_path = args.data_dir / f"features-{args.rows}x{args.width}-float32-global.bin"
    prepare_raw_partition(raw_path, 0, args.rows, args.width, args.overwrite)
    raw_paths = [raw_path]

    array_templates: dict[tuple[int, str], str] = {}
    suffix = "-consolidated" if args.consolidate else ""
    if "tiledb" in args.backends:
        for extent in args.tile_extents:
            if "rank" in args.array_layouts:
                template = str(
                    args.data_dir
                    / f"features-{args.rows}x{args.width}-tile-{extent}{suffix}-rank-{{rank}}-of-{args.world_size}.tdb"
                )
                array_templates[(extent, "rank")] = template
                for rank, (row_count, row_offset) in enumerate(
                    zip(partition, offsets, strict=True)
                ):
                    prepare_tiledb_array(
                        raw_path,
                        Path(template.replace("{rank}", str(rank))),
                        row_offset,
                        row_count,
                        args.width,
                        extent,
                        args.consolidate,
                        args.overwrite,
                        input_row_offset=row_offset,
                    )
            if "node" in args.array_layouts:
                node_uri = str(
                    args.data_dir
                    / f"features-{args.rows}x{args.width}-tile-{extent}{suffix}-node.tdb"
                )
                array_templates[(extent, "node")] = node_uri
                prepare_tiledb_array(
                    raw_path,
                    Path(node_uri),
                    0,
                    args.rows,
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
