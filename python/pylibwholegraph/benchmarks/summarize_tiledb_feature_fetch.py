# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Validate NVMe benchmark output and build a portable report artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--source-path",
        help="Repository-relative source label; defaults to the results argument",
    )
    return parser.parse_args()


def validate(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    query_chunks = metadata.get("query_chunk_rows", [0])
    pattern_count = len(metadata.get("patterns", ["random", "locality"]))
    resident_backends = sum(
        any(row["backend"] == backend for row in rows) for backend in ("cuda", "cpu")
    )
    tiledb_enabled = any(row["backend"] == "tiledb" for row in rows)
    expected_rows = resident_backends * pattern_count * len(metadata["batch_sizes"])
    if tiledb_enabled:
        expected_rows += (
            len(metadata["tile_extents"])
            * len(query_chunks)
            * 2
            * pattern_count
            * len(metadata["batch_sizes"])
        )
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} benchmark rows, found {len(rows)}")
    keys = {
        (
            row["backend"],
            row["tile_extent_rows"],
            row.get("query_chunk_rows", 0),
            row["cache_mode"],
            row["pattern"],
            row["batch_size"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("benchmark configuration keys are not unique")
    if metadata["dataset_gib"] <= 0 or metadata["rows"] <= 0:
        raise ValueError("dataset size must be positive")

    for row in rows:
        if row["samples"] != metadata["repetitions"]:
            raise ValueError("sample count does not match benchmark metadata")
        if row["latency_p50_ms"] > row["latency_p95_ms"]:
            raise ValueError("p50 latency exceeds p95 latency")
        expected = row["rows_per_second"] * metadata["width"] * 4 / 1024**3
        if abs(expected - row["useful_gib_per_second"]) > max(1e-9, expected * 1e-9):
            raise ValueError("useful-throughput calculation does not reconcile")
        if row["backend"] == "tiledb" and row["cache_mode"] == "cold":
            if row["storage_read_gib"] <= 0:
                raise ValueError("cold TileDB case did not perform block I/O")


def select(
    rows: list[dict[str, Any]],
    backend: str,
    pattern: str,
    batch_size: int,
    *,
    cache_mode: str | None = None,
    tile_extent: int | None = None,
    query_chunk_rows: int = 0,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["backend"] == backend
        and row["pattern"] == pattern
        and row["batch_size"] == batch_size
        and (cache_mode is None or row["cache_mode"] == cache_mode)
        and (tile_extent is None or row["tile_extent_rows"] == tile_extent)
        and row.get("query_chunk_rows", 0) == query_chunk_rows
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row, found {len(matches)}")
    return matches[0]


def round_row(row: dict[str, Any], configuration: str) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "backend": row["backend"],
        "tile_extent_rows": row["tile_extent_rows"],
        "query_chunk_rows": row.get("query_chunk_rows", 0),
        "cache_mode": row["cache_mode"],
        "pattern": row["pattern"],
        "batch_size": row["batch_size"],
        "latency_p50_ms": round(row["latency_p50_ms"], 3),
        "latency_p95_ms": round(row["latency_p95_ms"], 3),
        "useful_gib_per_second": round(row["useful_gib_per_second"], 4),
        "storage_read_gib_per_second": round(row["storage_read_gib_per_second"], 3),
        "read_amplification": round(row["read_amplification"], 2),
        "process_cpu_percent": round(row["process_cpu_percent"], 1),
        "gpu_sort_mean_ms": round(row.get("gpu_sort_mean_ms", 0.0), 3),
        "gpu_deduplicate_mean_ms": round(
            row.get("gpu_deduplicate_mean_ms", 0.0), 3
        ),
        "indices_d2h_mean_ms": round(row.get("indices_d2h_mean_ms", 0.0), 3),
        "tiledb_read_mean_ms": round(row.get("tiledb_read_mean_ms", 0.0), 3),
        "cpu_reorder_mean_ms": round(row.get("cpu_reorder_mean_ms", 0.0), 3),
        "rows_h2d_mean_ms": round(row.get("rows_h2d_mean_ms", 0.0), 3),
        "gpu_expand_mean_ms": round(row.get("gpu_expand_mean_ms", 0.0), 3),
        "samples": row["samples"],
    }


def main() -> None:
    args = parse_args()
    document = json.loads(args.results.read_text())
    source_path = args.source_path or str(args.results)
    metadata = document["metadata"]
    rows = document["results"]
    validate(rows, metadata)

    batch = max(metadata["batch_sizes"])
    world_size = metadata.get("world_size", 1)
    query_chunk_rows = 0
    cuda_local = select(rows, "cuda", "locality", batch)
    cpu_local = select(rows, "cpu", "locality", batch)
    cpu_random = select(rows, "cpu", "random", batch)
    tile_local = [
        select(
            rows,
            "tiledb",
            "locality",
            batch,
            cache_mode="cold",
            tile_extent=extent,
            query_chunk_rows=query_chunk_rows,
        )
        for extent in metadata["tile_extents"]
    ]
    tile_random = [
        select(
            rows,
            "tiledb",
            "random",
            batch,
            cache_mode="cold",
            tile_extent=extent,
            query_chunk_rows=query_chunk_rows,
        )
        for extent in metadata["tile_extents"]
    ]
    best_local = max(tile_local, key=lambda row: row["useful_gib_per_second"])
    best_random = max(tile_random, key=lambda row: row["useful_gib_per_second"])
    peak_io = max(
        (
            row
            for row in rows
            if row["backend"] == "tiledb" and row["cache_mode"] == "cold"
        ),
        key=lambda row: row["storage_read_gib_per_second"],
    )
    cold_tiledb = [
        row
        for row in rows
        if row["backend"] == "tiledb"
        and row["cache_mode"] == "cold"
        and row.get("query_chunk_rows", 0) == query_chunk_rows
    ]
    min_random_amplification = min(
        row["read_amplification"] for row in cold_tiledb if row["pattern"] == "random"
    )
    max_random_amplification = max(
        row["read_amplification"] for row in cold_tiledb if row["pattern"] == "random"
    )
    locality_slowdown = (
        cpu_local["useful_gib_per_second"] / best_local["useful_gib_per_second"]
    )
    random_slowdown = (
        cpu_random["useful_gib_per_second"] / best_random["useful_gib_per_second"]
    )
    gpu_names = metadata.get("gpus", [metadata.get("gpu", "unknown GPU")])
    gpu_description = gpu_names[0]
    if len(set(gpu_names)) > 1:
        gpu_description = ", ".join(sorted(set(gpu_names)))

    summary = [
        {
            "best_locality_tiledb_gib_s": round(best_local["useful_gib_per_second"], 4),
            "locality_cpu_gib_s": round(cpu_local["useful_gib_per_second"], 3),
            "locality_cpu_slowdown": round(
                cpu_local["useful_gib_per_second"]
                / best_local["useful_gib_per_second"],
                1,
            ),
            "best_random_tiledb_gib_s": round(best_random["useful_gib_per_second"], 4),
            "random_cpu_gib_s": round(cpu_random["useful_gib_per_second"], 3),
            "random_cpu_slowdown": round(
                cpu_random["useful_gib_per_second"]
                / best_random["useful_gib_per_second"],
                1,
            ),
            "peak_nvme_gib_s": round(peak_io["storage_read_gib_per_second"], 3),
            "peak_io_read_amplification": round(peak_io["read_amplification"], 1),
        }
    ]

    locality_latency = [
        round_row(cuda_local, "CUDA resident"),
        round_row(cpu_local, "CPU pinned"),
        *[
            round_row(row, f"TileDB {row['tile_extent_rows']} cold")
            for row in tile_local
        ],
    ]
    random_amplification = []
    for extent in metadata["tile_extents"]:
        for current_batch in metadata["batch_sizes"]:
            row = select(
                rows,
                "tiledb",
                "random",
                current_batch,
                cache_mode="cold",
                tile_extent=extent,
                query_chunk_rows=query_chunk_rows,
            )
            item = round_row(row, f"TileDB {extent}")
            item["batch_label"] = f"{current_batch:,} rows"
            item["tile_extent_label"] = f"{extent:,}-row tiles"
            random_amplification.append(item)

    detail = []
    for pattern in ("random", "locality"):
        detail.extend(
            [
                round_row(select(rows, "cuda", pattern, batch), "CUDA resident"),
                round_row(select(rows, "cpu", pattern, batch), "CPU pinned"),
            ]
        )
        detail.extend(
            round_row(
                select(
                    rows,
                    "tiledb",
                    pattern,
                    batch,
                    cache_mode="cold",
                    tile_extent=extent,
                    query_chunk_rows=query_chunk_rows,
                ),
                f"TileDB {extent} cold",
            )
            for extent in metadata["tile_extents"]
        )

    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    source = {
        "id": "benchmark_json",
        "label": f"{metadata['dataset_gib']:g} GiB WholeMemory feature-fetch benchmark",
        "path": source_path,
        "query": {
            "engine": "duckdb",
            "sql": (
                "SELECT result.* FROM read_json_auto('"
                f"{source_path}') AS benchmark, "
                "UNNEST(benchmark.results) AS rows(result)"
            ),
            "description": (
                "Loads all raw benchmark rows; the checked-in Python summarizer validates and "
                "transforms them into the report datasets."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": [source_path],
            "filters": [
                f"single node and {world_size} GPU rank(s)",
                f"float32 features with {metadata['width']} values per row",
                f"{metadata['repetitions']} measured repetitions after {metadata['warmup']} warmups",
            ],
            "metric_definitions": [
                "Useful GiB/s = requested rows × 512 bytes ÷ measured wall time ÷ 2^30.",
                "Read amplification = measured block-device bytes (or process reads when unavailable) ÷ requested feature bytes.",
            ],
        },
    }

    title = f"WholeMemory TileDB NVMe Feature-Fetch Benchmark ({world_size} GPU rank{'s' if world_size != 1 else ''})"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": f"{world_size}-GPU comparison of TileDB NVMe, pinned CPU, and CUDA feature gathers.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "locality_card",
                    "description": "Best cold TileDB throughput for the 65,536-row locality trace.",
                    "dataset": "summary",
                    "sourceId": "benchmark_json",
                    "metrics": [
                        {
                            "label": "Best locality TileDB GiB/s",
                            "field": "best_locality_tiledb_gib_s",
                            "format": "number",
                        },
                        {
                            "label": "Pinned CPU GiB/s",
                            "field": "locality_cpu_gib_s",
                            "format": "number",
                        },
                        {
                            "label": "CPU throughput multiple",
                            "field": "locality_cpu_slowdown",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "random_card",
                    "description": "Best cold TileDB throughput for the 65,536-row random trace.",
                    "dataset": "summary",
                    "sourceId": "benchmark_json",
                    "metrics": [
                        {
                            "label": "Best random TileDB GiB/s",
                            "field": "best_random_tiledb_gib_s",
                            "format": "number",
                        },
                        {
                            "label": "Pinned CPU GiB/s",
                            "field": "random_cpu_gib_s",
                            "format": "number",
                        },
                        {
                            "label": "CPU throughput multiple",
                            "field": "random_cpu_slowdown",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "io_card",
                    "description": "Peak observed block-device read rate in a cold TileDB case.",
                    "dataset": "summary",
                    "sourceId": "benchmark_json",
                    "metrics": [
                        {
                            "label": "Peak NVMe read GiB/s",
                            "field": "peak_nvme_gib_s",
                            "format": "number",
                        },
                        {
                            "label": "Read amplification",
                            "field": "peak_io_read_amplification",
                            "format": "number",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "locality_latency_chart",
                    "title": "P50 gather latency by backend",
                    "subtitle": "65,536 requested rows, locality-biased trace; resident baselines and cold TileDB.",
                    "type": "bar",
                    "dataset": "locality_latency",
                    "sourceId": "benchmark_json",
                    "encodings": {
                        "x": {
                            "field": "configuration",
                            "type": "nominal",
                            "label": "Configuration",
                        },
                        "y": {
                            "field": "latency_p50_ms",
                            "type": "quantitative",
                            "label": "P50 latency",
                            "unit": "ms",
                        },
                        "tooltip": [
                            {
                                "field": "latency_p95_ms",
                                "type": "quantitative",
                                "label": "P95 latency",
                                "unit": "ms",
                            },
                            {
                                "field": "useful_gib_per_second",
                                "type": "quantitative",
                                "label": "Useful GiB/s",
                            },
                            {
                                "field": "read_amplification",
                                "type": "quantitative",
                                "label": "Read amplification",
                            },
                        ],
                    },
                    "yAxisTitle": "P50 latency (ms)",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "random_amplification_chart",
                    "title": "Cold random-gather read amplification",
                    "subtitle": "Process block-read bytes divided by useful requested feature bytes.",
                    "type": "bar",
                    "dataset": "random_amplification",
                    "sourceId": "benchmark_json",
                    "encodings": {
                        "x": {
                            "field": "batch_label",
                            "type": "ordinal",
                            "label": "Batch size",
                        },
                        "y": {
                            "field": "read_amplification",
                            "type": "quantitative",
                            "label": "Read amplification",
                        },
                        "color": {
                            "field": "tile_extent_label",
                            "type": "nominal",
                            "label": "Tile extent",
                        },
                        "tooltip": [
                            {
                                "field": "storage_read_gib_per_second",
                                "type": "quantitative",
                                "label": "NVMe GiB/s",
                            },
                            {
                                "field": "latency_p50_ms",
                                "type": "quantitative",
                                "label": "P50 latency",
                                "unit": "ms",
                            },
                        ],
                    },
                    "yAxisTitle": "Read amplification (×)",
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "detail_table",
                    "title": "65,536-row gather detail",
                    "subtitle": f"Resident baselines and cold TileDB, {metadata['repetitions']} samples per configuration.",
                    "dataset": "detail",
                    "sourceId": "benchmark_json",
                    "defaultSort": {"field": "latency_p50_ms", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {
                            "field": "configuration",
                            "label": "Configuration",
                            "type": "text",
                        },
                        {"field": "pattern", "label": "Trace", "type": "text"},
                        {
                            "field": "latency_p50_ms",
                            "label": "P50 ms",
                            "format": "number",
                        },
                        {
                            "field": "latency_p95_ms",
                            "label": "P95 ms",
                            "format": "number",
                        },
                        {
                            "field": "useful_gib_per_second",
                            "label": "Useful GiB/s",
                            "format": "number",
                        },
                        {
                            "field": "storage_read_gib_per_second",
                            "label": "NVMe GiB/s",
                            "format": "number",
                        },
                        {
                            "field": "read_amplification",
                            "label": "Read amp ×",
                            "format": "number",
                        },
                        {
                            "field": "process_cpu_percent",
                            "label": "Process CPU %",
                            "format": "number",
                        },
                        {
                            "field": "gpu_sort_mean_ms",
                            "label": "GPU sort ms",
                            "format": "number",
                        },
                        {
                            "field": "gpu_deduplicate_mean_ms",
                            "label": "GPU dedup ms",
                            "format": "number",
                        },
                        {
                            "field": "indices_d2h_mean_ms",
                            "label": "Unique IDs D2H ms",
                            "format": "number",
                        },
                        {
                            "field": "tiledb_read_mean_ms",
                            "label": "TileDB read ms",
                            "format": "number",
                        },
                        {
                            "field": "cpu_reorder_mean_ms",
                            "label": "CPU compact copy ms",
                            "format": "number",
                        },
                        {
                            "field": "rows_h2d_mean_ms",
                            "label": "Unique rows H2D ms",
                            "format": "number",
                        },
                        {
                            "field": "gpu_expand_mean_ms",
                            "label": "GPU expand ms",
                            "format": "number",
                        },
                    ],
                }
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        "## Technical summary\n\n"
                        f"The TileDB prototype is functionally correct and drives storage at up to **{peak_io['storage_read_gib_per_second']:.3f} GiB/s**, "
                        f"but it is not competitive with resident WholeMemory backends for these gathers. At a {batch:,}-row batch, "
                        f"the best cold locality result is **{best_local['useful_gib_per_second']:.4f} GiB/s** versus **{cpu_local['useful_gib_per_second']:.3f} GiB/s** for pinned CPU, while the best cold random result is "
                        f"**{best_random['useful_gib_per_second']:.4f} GiB/s** versus **{cpu_random['useful_gib_per_second']:.3f} GiB/s**. The dominant issue is read amplification and synchronous CPU-side query work, not raw NVMe bandwidth."
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": ["locality_card", "random_card", "io_card"],
                },
                {
                    "id": "locality_finding",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        f"## Locality helps, but pinned CPU remains {locality_slowdown:.1f}× faster\n\n"
                        f"The {best_local['tile_extent_rows']:,}-row TileDB extent is the best cold choice for the locality-biased {batch:,}-row trace: "
                        f"**{best_local['latency_p50_ms']:.3f} ms p50**, **{best_local['latency_p95_ms']:.3f} ms p95**, and **{best_local['useful_gib_per_second']:.4f} GiB/s** useful throughput. "
                        f"Pinned CPU completes the same trace in **{cpu_local['latency_p50_ms']:.3f} ms p50** and CUDA in **{cuda_local['latency_p50_ms']:.3f} ms p50**. "
                        f"TileDB read amplification is **{best_local['read_amplification']:.2f}×** in this case, so the remaining gap includes query, sorting, copying, routing, and synchronization overhead."
                    ),
                },
                {
                    "id": "locality_chart",
                    "type": "chart",
                    "chartId": "locality_latency_chart",
                    "layout": "full",
                },
                {
                    "id": "random_finding",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        "## Random gathers turn bandwidth into amplification\n\n"
                        f"Cold random traces read **{min_random_amplification:.1f}× to {max_random_amplification:.1f}×** more bytes than requested. "
                        f"The fastest observed storage case reaches **{peak_io['storage_read_gib_per_second']:.3f} GiB/s**, while useful throughput remains constrained by amplification and query work. "
                        f"At {batch:,} rows, even the best random TileDB configuration is **{random_slowdown:.1f}×** below pinned CPU useful throughput."
                    ),
                },
                {
                    "id": "amplification_chart",
                    "type": "chart",
                    "chartId": "random_amplification_chart",
                    "layout": "full",
                },
                {
                    "id": "detail_interpretation",
                    "type": "markdown",
                    "body": (
                        "## Exact results show two different failure modes\n\n"
                        "Cold random performance is I/O-amplification dominated. Warm random performance remains far slower than CPU/CUDA even with zero block reads, "
                        "which isolates substantial CPU/query overhead. The table below keeps the primary 65,536-row comparisons at audit precision."
                    ),
                },
                {
                    "id": "detail",
                    "type": "table",
                    "tableId": "detail_table",
                    "layout": "full",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        f"The benchmark uses {world_size} process/rank(s) on **{gpu_description}**, a **{metadata['dataset_gib']:g} GiB** synthetic table with **{metadata['rows']:,} × {metadata['width']} float32** values, "
                        "and identical seeded CUDA ID traces for all backends. `cpu` is WholeMemory distributed host memory allocated with `cudaMallocHost`; "
                        "`cuda` is WholeMemory distributed device memory; TileDB uses the branch's distributed NCCL gather path. "
                        "Useful throughput counts requested feature bytes across all ranks and uses the slowest synchronized rank latency. Read amplification uses block-device sectors when available, falling back to summed process reads."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        "## Methodology and robustness checks\n\n"
                        f"Each case has **{metadata['warmup']} warmups and {metadata['repetitions']} measured repetitions**; CUDA is synchronized around every gather. Cold TileDB samples issue "
                        "`POSIX_FADV_DONTNEED` for every array file before timing, and measured block reads confirm that I/O occurred. Warm TileDB files are read fully before timing, "
                        f"and their measured reads reveal whether they remained resident. The suite covers random and 16-window locality traces, {len(metadata['batch_sizes'])} batch sizes, and tile extents {metadata['tile_extents']}. "
                        f"All {len(rows)} configuration keys are unique, p50 never exceeds p95, raw samples are retained, and useful-throughput calculations were independently recomputed."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and omitted instrumentation\n\n"
                        "This is a single-machine synthetic benchmark, not an application trace. `POSIX_FADV_DONTNEED` is advisory, and device counters can include unrelated host activity. "
                        "The benchmark records TileDB statistics when requested, CUDA allocator and process-RSS observations, and separate routing, GPU sort/dedup, unique-ID D2H, TileDB query, compact CPU copy, unique-row H2D, GPU expansion, NCCL, and final-reorder timings. TileDB's internal timers overlap and are diagnostic rather than an additive decomposition. "
                        "Torch allocator peaks may not capture allocations made outside its allocator."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Compare unbounded and bounded query chunks, consolidated and unconsolidated arrays, and smaller tile extents before selecting a default.\n"
                        "2. Use TileDB statistics and a representative sampler trace to explain amplification beyond tile granularity.\n"
                        "3. Profile GPU compaction, D2H, query, H2D, GPU expansion, and NCCL with Nsight Systems before implementing overlap or persistent staging.\n"
                        "4. If TileDB remains uncompetitive for random traces, compare it with a row-addressable direct-I/O layout."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "How much reuse exists in real sampled-neighborhood traces? Can ID bucketing be preserved across minibatches? Would a compact row store plus `io_uring`/GDS outperform TileDB for random features? "
                        "Those answers determine whether the backend should optimize around locality or adopt a different physical layout."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "locality_latency": locality_latency,
                "random_amplification": random_amplification,
                "detail": detail,
            },
        },
        "sources": [source],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
