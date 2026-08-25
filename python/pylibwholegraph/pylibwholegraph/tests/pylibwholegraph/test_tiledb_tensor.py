# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch as wgth
from pylibwholegraph.utils.multiprocess import multiprocess_run


def tiledb_test_enabled():
    return os.getenv("TEST_WM_TILEDB", "").lower() in {"1", "true", "on"}


def tiledb_routine(world_rank: int, world_size: int):
    torch = pytest.importorskip("torch")
    comm, _ = wgth.init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size, wm_log_level="error"
    )

    shared_root = os.getenv("TEST_WM_TILEDB_ROOT")
    root_context = (
        nullcontext(shared_root)
        if shared_root is not None
        else tempfile.TemporaryDirectory(prefix="wholememory_tiledb_python_")
    )
    with root_context as root:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        rows_per_rank = 5
        total_rows = rows_per_rank * world_size
        first_value = world_rank * rows_per_rank * 2
        values = np.arange(
            first_value, first_value + rows_per_rank * 2, dtype=np.float32
        ).reshape(rows_per_rank, 2)
        raw_path = root / f"features-{world_rank}.bin"
        array_path = root / f"rank_{world_rank}.tdb"
        values.tofile(raw_path)
        subprocess.run(
            [
                "wholememory_tiledb_ingest",
                str(array_path),
                str(raw_path),
                str(rows_per_rank),
                str(values[0].nbytes),
                "2",
                "5",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        comm.barrier()

        node_array_path = root / "node.tdb"
        if world_rank == 0:
            node_raw_path = root / "node-features.bin"
            np.arange(total_rows * 2, dtype=np.float32).tofile(node_raw_path)
            subprocess.run(
                [
                    "wholememory_tiledb_ingest",
                    str(node_array_path),
                    str(node_raw_path),
                    str(total_rows),
                    str(values[0].nbytes),
                    "2",
                    "5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        comm.barrier()

        tensor = wgth.create_wholememory_tensor_from_tiledb(
            comm,
            str(root / "rank_{rank}.tdb"),
            [total_rows, 2],
            torch.float32,
            tensor_entry_partition=[rows_per_rank] * world_size,
        )
        requested_rows = [rank * rows_per_rank for rank in range(world_size)]
        requested_rows.extend([total_rows - 1, 0])
        indices = torch.tensor(requested_rows, dtype=torch.int64, device="cuda")
        global_values = torch.arange(total_rows * 2, dtype=torch.float32).reshape(
            total_rows, 2
        )
        gathered = tensor.gather(indices)
        torch.testing.assert_close(
            gathered.cpu(),
            global_values[indices.cpu()],
            rtol=0,
            atol=0,
        )
        metrics = wmb.get_last_tiledb_gather_metrics()
        assert metrics["valid"]
        assert metrics["index_bytes"] > 0
        assert metrics["raw_staging_bytes"] > 0
        assert metrics["output_bytes"] > 0
        assert metrics["tiledb_read_ms"] >= 0
        assert metrics["id_sort_ms"] >= 0
        assert metrics["range_build_ms"] >= 0
        assert metrics["query_submit_ms"] >= 0
        assert metrics["cpu_reorder_ms"] >= 0
        assert metrics["storage_requested_rows"] > 0
        assert metrics["storage_unique_rows"] > 0
        assert metrics["storage_range_count"] > 0
        assert metrics["storage_query_count"] > 0
        with pytest.raises(NotImplementedError, match="^Not supported$"):
            tensor.get_local_tensor()

        invalid_partitions = (
            [rows_per_rank - 1] + [rows_per_rank] * (world_size - 1),
            [0] + [rows_per_rank] * (world_size - 1),
            [rows_per_rank] * (world_size + 1),
        )
        for partition in invalid_partitions:
            with pytest.raises(ValueError):
                wgth.create_wholememory_tensor_from_tiledb(
                    comm,
                    str(root / "rank_{rank}.tdb"),
                    [total_rows, 2],
                    torch.float32,
                    tensor_entry_partition=partition,
                )

        wgth.destroy_wholememory_tensor(tensor)

        node_tensor = wgth.create_wholememory_tensor_from_tiledb(
            comm,
            str(node_array_path),
            [total_rows, 2],
            torch.float32,
            tensor_entry_partition=[rows_per_rank] * world_size,
            array_layout="node",
        )
        node_gathered = node_tensor.gather(indices)
        torch.testing.assert_close(
            node_gathered.cpu(),
            global_values[indices.cpu()],
            rtol=0,
            atol=0,
        )
        wgth.destroy_wholememory_tensor(node_tensor)

    wgth.finalize()


@pytest.mark.skipif(not tiledb_test_enabled(), reason="TEST_WM_TILEDB is not enabled")
@pytest.mark.skipif(
    shutil.which("wholememory_tiledb_ingest") is None,
    reason="wholememory_tiledb_ingest is not installed",
)
def test_tiledb_list_partition_and_local_mapping():
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    world_size = int(os.getenv("TEST_WM_TILEDB_WORLD_SIZE", "1"))
    assert 0 < world_size <= gpu_count
    with tempfile.TemporaryDirectory(
        prefix="wholememory_tiledb_python_shared_"
    ) as root:
        os.environ["TEST_WM_TILEDB_ROOT"] = root
        try:
            multiprocess_run(world_size, tiledb_routine, inline_single_process=True)
        finally:
            os.environ.pop("TEST_WM_TILEDB_ROOT", None)
