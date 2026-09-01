#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Validate the focused TabICLv2 overlap benchmark result directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    "tabicl-overlap-clustered-width-2048.json": (42, 210),
    "tabicl-overlap-scattered-width-2048.json": (6, 30),
    "tabicl-overlap-multirun-width-2048.json": (18, 90),
    "tabicl-continuity-width-2048.json": (6, 30),
}
PHASES = (
    "id_routing",
    "gpu_sort",
    "gpu_deduplicate",
    "staging_allocation",
    "indices_d2h",
    "tiledb_read",
    "id_decode",
    "id_sort",
    "id_deduplicate",
    "query_setup",
    "range_build",
    "query_submit",
    "cpu_reorder",
    "rows_h2d",
    "gpu_expand",
    "embedding_exchange",
    "output_reorder",
)
PHASE_COUNTS = (
    "storage_requested_rows",
    "storage_unique_rows",
    "storage_range_count",
    "storage_query_count",
    "index_bytes",
    "raw_staging_bytes",
    "output_bytes",
)
PAIRED_CASES = {"cross_rank_25", "within_rank_25"}
IOPS_FIELDS = (
    "storage_read_ops",
    "storage_write_ops",
    "storage_total_io_ops",
    "storage_read_iops",
    "storage_write_iops",
    "storage_iops",
    "storage_total_iops",
)


def pairing_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sample["backend"],
        sample["array_layout"],
        sample["tile_extent_rows"],
        sample["query_chunk_rows"],
        sample["cache_mode"],
        sample["batch_size"],
        sample["overlap_placement"],
        sample["sample"],
    )


def validate_file(path: Path, expected: tuple[int, int]) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    document = json.loads(path.read_text())
    results = document["results"]
    samples = document["samples"]
    if (len(results), len(samples)) != expected:
        raise AssertionError(
            f"{path.name}: expected {expected}, got {(len(results), len(samples))}"
        )
    for result in results:
        if (
            not math.isfinite(result["latency_mean_ms"])
            or result["latency_mean_ms"] <= 0
        ):
            raise AssertionError(f"{path.name}: invalid aggregate latency")
        for field in IOPS_FIELDS:
            value = result.get(field)
            if value is None or not math.isfinite(float(value)) or value < 0:
                raise AssertionError(f"{path.name}: invalid aggregate {field}")
        for phase in PHASES:
            if f"{phase}_rank_max_mean_ms" not in result:
                raise AssertionError(
                    f"{path.name}: missing rank-aware {phase} aggregate"
                )
        for count in PHASE_COUNTS:
            if f"{count}_rank_max_mean" not in result:
                raise AssertionError(
                    f"{path.name}: missing rank-aware {count} aggregate"
                )
    for sample in samples:
        if not math.isfinite(sample["latency_ms"]) or sample["latency_ms"] <= 0:
            raise AssertionError(f"{path.name}: invalid sample latency")
        for field in IOPS_FIELDS:
            value = sample.get(field)
            if value is None or not math.isfinite(float(value)) or value < 0:
                raise AssertionError(f"{path.name}: invalid sample {field}")
        for field in ("slowest_rank", "slowest_rank_has_storage", "storage_owner_count"):
            if field not in sample:
                raise AssertionError(f"{path.name}: missing {field}")
        for phase in PHASES:
            for suffix in (
                "rank_max_ms",
                "rank_mean_ms",
                "rank_max_rank",
                "storage_owner_max_ms",
                "storage_owner_mean_ms",
            ):
                if f"{phase}_{suffix}" not in sample:
                    raise AssertionError(
                        f"{path.name}: missing {phase}_{suffix}"
                    )
        for count in PHASE_COUNTS:
            for suffix in (
                "rank_max",
                "rank_mean",
                "rank_max_rank",
                "storage_owner_max",
                "storage_owner_mean",
            ):
                if f"{count}_{suffix}" not in sample:
                    raise AssertionError(
                        f"{path.name}: missing {count}_{suffix}"
                    )
    return document


def validate_pairs(document: dict[str, Any], label: str) -> None:
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for sample in document["samples"]:
        case = sample["overlap_case"]
        if case not in PAIRED_CASES:
            continue
        digest = sample["node_unique_id_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise AssertionError(f"{label}: invalid node unique-ID digest")
        pairs.setdefault(pairing_key(sample), {})[case] = sample
    for key, cases in pairs.items():
        if set(cases) != PAIRED_CASES:
            raise AssertionError(f"{label}: incomplete topology pair for {key}")
        cross = cases["cross_rank_25"]
        within = cases["within_rank_25"]
        if cross["node_unique_id_sha256"] != within["node_unique_id_sha256"]:
            raise AssertionError(f"{label}: topology pair changed the unique ID set")
        if (
            cross["node_contiguous_ranges"] != within["node_contiguous_ranges"]
            or cross["node_estimated_tiles_touched"]
            != within["node_estimated_tiles_touched"]
        ):
            raise AssertionError(f"{label}: topology pair changed physical placement")


def validate_multirun(document: dict[str, Any]) -> None:
    for sample in document["samples"]:
        expected_runs = int(
            sample["overlap_placement"].removeprefix("clustered_runs_")
        )
        if sample["clustered_run_count"] != expected_runs:
            raise AssertionError("multi-run placement recorded the wrong run count")
        if sample["node_contiguous_ranges"] != expected_runs:
            raise AssertionError(
                f"expected {expected_runs} node ranges, got "
                f"{sample['node_contiguous_ranges']}"
            )
        if sample["owner_unique_max_to_mean"] > 1.01:
            raise AssertionError("multi-run unique rows are not balanced across owners")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    documents = {
        name: validate_file(args.result_dir / name, counts)
        for name, counts in EXPECTED.items()
    }
    validate_pairs(
        documents["tabicl-overlap-scattered-width-2048.json"], "scattered"
    )
    validate_pairs(
        documents["tabicl-overlap-multirun-width-2048.json"], "multi-run"
    )
    validate_multirun(documents["tabicl-overlap-multirun-width-2048.json"])
    aggregate_count = sum(len(doc["results"]) for doc in documents.values())
    sample_count = sum(len(doc["samples"]) for doc in documents.values())
    print(
        f"PASS: {aggregate_count} aggregate configurations, "
        f"{sample_count} measured samples, paired traces and rank-aware phases valid"
    )


if __name__ == "__main__":
    main()
