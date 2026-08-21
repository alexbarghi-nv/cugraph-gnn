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
        values = np.arange(10, dtype=np.float32).reshape(5, 2)
        raw_path = root / "features.bin"
        array_path = root / "rank_0.tdb"
        values.tofile(raw_path)
        subprocess.run(
            [
                "wholememory_tiledb_ingest",
                str(array_path),
                str(raw_path),
                "5",
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
            [5, 2],
            torch.float32,
            tensor_entry_partition=[5],
        )
        indices = torch.tensor([4, 0, 4, 2], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(
            tensor.gather(indices).cpu(),
            torch.from_numpy(values)[indices.cpu()],
            rtol=0,
            atol=0,
        )
        with pytest.raises(NotImplementedError, match="^Not supported$"):
            tensor.get_local_tensor()

        for partition in ([4], [0], [2, 3]):
            with pytest.raises(ValueError):
                wgth.create_wholememory_tensor_from_tiledb(
                    comm,
                    str(root / "rank_{rank}.tdb"),
                    [5, 2],
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
    assert wmb.fork_get_gpu_count() > 0
    multiprocess_run(1, tiledb_routine, inline_single_process=True)
