#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Validate the focused Direct I/O and IOPS benchmark result directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BUFFERED_EXPECTED = {
    "iops-overlap-clustered-width-2048.json": (21, 105),
    "iops-overlap-scattered-width-2048.json": (6, 30),
    "iops-overlap-multirun-width-2048.json": (18, 90),
}
IOPS_FIELDS = (
    "storage_read_ops",
    "storage_write_ops",
    "storage_total_io_ops",
    "storage_read_iops",
    "storage_write_iops",
    "storage_iops",
    "storage_total_iops",
)
DIRECT_FIELDS = (
    "direct_io_open_attempts",
    "direct_io_open_successes",
    "direct_io_open_failures",
    "direct_io_read_ops",
    "direct_io_requested_bytes",
    "direct_io_submitted_bytes",
    "direct_io_returned_bytes",
    "direct_io_read_failures",
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    return json.loads(path.read_text())


def validate_iops(document: dict[str, Any], label: str) -> None:
    for collection in (document["results"], document["samples"]):
        for row in collection:
            for field in IOPS_FIELDS:
                value = row.get(field)
                if value is None or not math.isfinite(float(value)) or value < 0:
                    raise AssertionError(f"{label}: invalid {field}")
            if row["storage_total_io_ops"] != (
                row["storage_read_ops"] + row["storage_write_ops"]
            ):
                raise AssertionError(f"{label}: total I/O operation count mismatch")


def validate_direct(document: dict[str, Any]) -> None:
    if not document["metadata"].get("tiledb_direct_io"):
        raise AssertionError("Direct I/O metadata is not enabled")
    if not document["metadata"].get("tiledb_direct_io_preload_available"):
        raise AssertionError("Direct I/O preload was not detected")
    if len(document["results"]) not in (1, 3):
        raise AssertionError("expected the primary Direct I/O case or its sensitivity set")
    if len(document["samples"]) != len(document["results"]) * 5:
        raise AssertionError("unexpected Direct I/O sample count")
    for row in document["results"]:
        if row["cache_mode"] != "direct":
            raise AssertionError("Direct I/O row has the wrong cache mode")
        for field in DIRECT_FIELDS:
            if row.get(field) is None:
                raise AssertionError(f"Direct I/O row is missing {field}")
        if row["direct_io_open_successes"] <= 0 or row["direct_io_read_ops"] <= 0:
            raise AssertionError("Direct I/O did not intercept file reads")
        if row["direct_io_open_failures"] != 0 or row["direct_io_read_failures"] != 0:
            raise AssertionError("Direct I/O reported failed opens or reads")
        if row["direct_io_submitted_bytes"] < row["direct_io_requested_bytes"]:
            raise AssertionError("Direct I/O aligned-byte accounting is inconsistent")
        if row["direct_io_returned_bytes"] != row["direct_io_requested_bytes"]:
            raise AssertionError("Direct I/O did not return every requested byte")
        if row["storage_read_ops"] <= 0:
            raise AssertionError("Direct I/O produced no physical device read operations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    aggregate_count = 0
    sample_count = 0
    for name, expected in BUFFERED_EXPECTED.items():
        document = load(args.result_dir / name)
        actual = (len(document["results"]), len(document["samples"]))
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        validate_iops(document, name)
        aggregate_count += actual[0]
        sample_count += actual[1]

    direct = load(args.result_dir / "direct-io-multirun-width-2048.json")
    validate_iops(direct, "direct-io-multirun-width-2048.json")
    validate_direct(direct)
    aggregate_count += len(direct["results"])
    sample_count += len(direct["samples"])
    print(
        f"PASS: {aggregate_count} aggregate configurations, {sample_count} "
        "samples, IOPS complete, Direct I/O interception verified"
    )


if __name__ == "__main__":
    main()
