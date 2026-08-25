# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
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

    with tempfile.TemporaryDirectory(prefix="wholememory_tiledb_python_") as root:
        root = Path(root)
        rows_per_rank = 5
        total_rows = rows_per_rank * world_size
        first_value = world_rank * rows_per_rank * 2
        values = np.arange(
            first_value, first_value + rows_per_rank * 2, dtype=np.float32
        ).reshape(rows_per_rank, 2)
        raw_path = root / "features.bin"
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
        assert metrics["cpu_reorder_ms"] >= 0
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
    multiprocess_run(world_size, tiledb_routine, inline_single_process=True)
