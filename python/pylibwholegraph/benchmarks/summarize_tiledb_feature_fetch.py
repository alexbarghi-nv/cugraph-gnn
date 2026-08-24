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
        default="nvme_benchmark_results/20260821-gb10/final-8g.json",
    )
    return parser.parse_args()


def validate(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    if len(rows) != 48:
        raise ValueError(f"expected 48 benchmark rows, found {len(rows)}")
    keys = {
        (
            row["backend"],
            row["tile_extent_rows"],
            row["cache_mode"],
            row["pattern"],
            row["batch_size"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("benchmark configuration keys are not unique")
    if metadata["dataset_gib"] != 8.0 or metadata["rows"] != 16_777_216:
        raise ValueError("this report expects the validated 8 GiB benchmark")

    for row in rows:
        if row["samples"] != metadata["repetitions"]:
            raise ValueError("sample count does not match benchmark metadata")
        if row["latency_p50_ms"] > row["latency_p95_ms"]:
            raise ValueError("p50 latency exceeds p95 latency")
        expected = row["rows_per_second"] * metadata["width"] * 4 / 1024**3
        if abs(expected - row["useful_gib_per_second"]) > max(1e-9, expected * 1e-9):
            raise ValueError("useful-throughput calculation does not reconcile")
        if row["backend"] in {"cuda", "cpu"} and row["storage_read_gib"] != 0:
            raise ValueError("resident backend unexpectedly performed block I/O")
        if row["backend"] == "tiledb" and row["cache_mode"] == "warm":
            if row["storage_read_gib"] != 0:
                raise ValueError("warm TileDB case unexpectedly performed block I/O")
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
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["backend"] == backend
        and row["pattern"] == pattern
        and row["batch_size"] == batch_size
        and (cache_mode is None or row["cache_mode"] == cache_mode)
        and (tile_extent is None or row["tile_extent_rows"] == tile_extent)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row, found {len(matches)}")
    return matches[0]


def round_row(row: dict[str, Any], configuration: str) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "backend": row["backend"],
        "tile_extent_rows": row["tile_extent_rows"],
        "cache_mode": row["cache_mode"],
        "pattern": row["pattern"],
        "batch_size": row["batch_size"],
        "latency_p50_ms": round(row["latency_p50_ms"], 3),
        "latency_p95_ms": round(row["latency_p95_ms"], 3),
        "useful_gib_per_second": round(row["useful_gib_per_second"], 4),
        "storage_read_gib_per_second": round(row["storage_read_gib_per_second"], 3),
        "read_amplification": round(row["read_amplification"], 2),
        "process_cpu_percent": round(row["process_cpu_percent"], 1),
        "samples": row["samples"],
    }


def main() -> None:
    args = parse_args()
    document = json.loads(args.results.read_text())
    metadata = document["metadata"]
    rows = document["results"]
    validate(rows, metadata)

    batch = 65_536
    cuda_local = select(rows, "cuda", "locality", batch)
    cpu_local = select(rows, "cpu", "locality", batch)
    cpu_random = select(rows, "cpu", "random", batch)
    tile_local = [
        select(rows, "tiledb", "locality", batch, cache_mode="cold", tile_extent=extent)
        for extent in metadata["tile_extents"]
    ]
    tile_random = [
        select(rows, "tiledb", "random", batch, cache_mode="cold", tile_extent=extent)
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
        "label": "8 GiB WholeMemory feature-fetch benchmark",
        "path": args.source_path,
        "query": {
            "engine": "duckdb",
            "sql": (
                "SELECT result.* FROM read_json_auto('"
                f"{args.source_path}') AS benchmark, "
                "UNNEST(benchmark.results) AS rows(result)"
            ),
            "description": (
                "Loads all raw benchmark rows; the checked-in Python summarizer validates and "
                "transforms them into the report datasets."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": [args.source_path],
            "filters": [
                "single node and single GPU",
                "float32 features with 128 values per row",
                "10 measured repetitions after 3 warmups",
            ],
            "metric_definitions": [
                "Useful GiB/s = requested rows × 512 bytes ÷ measured wall time ÷ 2^30.",
                "Read amplification = process block-read bytes ÷ requested feature bytes.",
            ],
        },
    }

    title = "WholeMemory TileDB NVMe Feature-Fetch Benchmark"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Single-GPU comparison of TileDB NVMe, pinned CPU, and CUDA feature gathers.",
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
                    "description": "Peak observed process block-read rate in a cold TileDB case.",
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
                    "subtitle": "Resident baselines and cold TileDB, 10 samples per configuration.",
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
                        "The TileDB prototype is functionally correct and drives the NVMe at up to **7.71 GiB/s**, "
                        "but it is not competitive with resident WholeMemory backends for these gathers. At a 65,536-row batch, "
                        "the best cold locality result is **0.525 GiB/s** versus **30.462 GiB/s** for pinned CPU, while the best cold random result is "
                        "**0.0111 GiB/s** versus **7.648 GiB/s**. The dominant issue is read amplification and synchronous CPU-side query work, not raw NVMe bandwidth."
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
                        "## Locality helps, but the resident backends remain roughly 58–62× faster\n\n"
                        "The 256-row TileDB extent is the best cold choice for the locality-biased 65,536-row trace: "
                        "**56.35 ms p50**, **76.37 ms p95**, and **0.525 GiB/s** useful throughput. "
                        "Pinned CPU completes the same trace in **1.025 ms p50** and CUDA in **0.959 ms p50**. "
                        "Locality keeps TileDB read amplification near **1.27×**, so most of the remaining gap is query, sorting, copying, and synchronization overhead."
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
                        "Cold random traces read **253× to 16,069×** more bytes than requested. The fastest disk case reaches **7.71 GiB/s**, "
                        "yet delivers only **0.0041 GiB/s** of requested features because its read amplification is **1,868×**. "
                        "At 65,536 rows, even the best random TileDB configuration is about **687×** below pinned CPU useful throughput."
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
                        "The benchmark uses one NVIDIA GB10, one process/rank, an **8 GiB** synthetic table with **16,777,216 × 128 float32** values, "
                        "and identical seeded CUDA ID traces for all backends. `cpu` is WholeMemory distributed host memory allocated with `cudaMallocHost`; "
                        "`cuda` is WholeMemory distributed device memory; TileDB uses the branch's distributed NCCL gather path. "
                        "Useful throughput counts only requested feature bytes. Read amplification divides Linux process block-read bytes by those useful bytes."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "benchmark_json",
                    "body": (
                        "## Methodology and robustness checks\n\n"
                        "Each case has **3 warmups and 10 measured repetitions**; CUDA is synchronized around every gather. Cold TileDB samples issue "
                        "`POSIX_FADV_DONTNEED` for every array file before timing, and measured block reads confirm that I/O occurred. Warm TileDB files are read fully before timing, "
                        "and every warm result records zero block reads. The suite covers random and 16-window locality traces, three batch sizes, and 256-, 4,096-, and 65,536-row tile extents. "
                        "All 48 configuration keys are unique, p50 never exceeds p95, and useful-throughput calculations were independently recomputed."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and omitted instrumentation\n\n"
                        "This is a single-machine synthetic benchmark, not an application trace. `POSIX_FADV_DONTNEED` is advisory, although process I/O counters verify cold reads. "
                        "Ten samples give a directional p95 rather than a production tail estimate. The current API does not expose TileDB internal statistics, isolated H2D bandwidth, "
                        "or peak pinned/device staging memory, so those runbook metrics remain unmeasured. `torch` allocator peaks would not capture WholeMemory's C++ allocations, and GB10 NVML memory reporting is unavailable on this host."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Add bounded chunking and overlap TileDB reads, H2D copies, and NCCL communication; the current synchronous full-batch staging path serializes them.\n"
                        "2. Reuse pinned buffers and query objects to reduce allocation and setup costs visible even in zero-I/O warm runs.\n"
                        "3. Replace tile-amplified random access with a row-addressable/direct-I/O layout or an adaptive small-tile path; preserve the 256-row extent for locality-biased workloads.\n"
                        "4. Expose TileDB stats and WholeMemory staging counters, then profile a representative GNN sampler trace with Nsight Systems before evaluating multi-GPU scale."
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
