# TileDB WholeGraph backend verification

Date: 2026-08-21
Branch: `prototype/tiledb-backend`
Host: `dgx19`

Updated: 2026-08-25 with the eight-GPU RTX PRO 6000 NVMe results.

## Executive summary

The TileDB-enabled library and Python packages build successfully in an isolated Conda environment.
One-rank and two-rank same-node gathers match the pinned-CPU WholeMemory backend exactly for the
tested matrix, including cross-rank IDs, duplicates, empty batches, uneven partition boundaries,
one-dimensional tensors, and subcolumn views. The C++ TileDB storage tests and the cuGraph-PyG
`DistTensor` smoke test pass.

The locally actionable checklist on `dgx19` is complete. True multi-node execution remains blocked
because that session had no scheduler allocation or second-node endpoint. The `dgx19` host also had
no NVMe device, but a subsequent single-GPU cold-NVMe benchmark completed on a DGX Spark with an
NVIDIA GB10. The enhanced benchmark then completed on a separate host with eight NVIDIA RTX PRO
6000 Blackwell Server Edition GPUs and local NVMe storage, including the unconsolidated and
consolidated sweeps described below.

Scatter, file load, and file store correctly return `WHOLEMEMORY_NOT_SUPPORTED`. A missing Cython
handler initially displayed this as `Error code 9 not recognized`; the handler was added and these
operations now raise `NotImplementedError("Not supported")`.

### Post-verification fixes

The branch now rejects direct local-memory access to TileDB handles with
`WHOLEMEMORY_NOT_SUPPORTED`. The direct pylibwholegraph constructor also validates an explicit
partition and converts ordinary Python lists to a `numpy.uintp` array before entering Cython. A
focused Python regression test was added, and both fixes passed the one-rank and two-rank GPU
verification matrices.

## Environment preparation

Testing was isolated from the existing `rapids` environment in:

```text
/raid/abarghi/.local/share/mamba/envs/tiledb-wg
```

The existing `rapids` prefix was left untouched after the request to stop restoring it.

The environment specification and exact Linux lock are:

- `conda/environments/tiledb-wg.yaml`
- `conda/environments/tiledb-wg-linux-64.lock`

Important installed versions were:

| Package | Installed version/build |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0, `cuda129_mkl_py312_hb9da02c_305` |
| libtorch | 2.10.0, `cuda129_mkl_hda1b8b5_305` |
| torchdata | 0.11.0 |
| TileDB C/C++ | 2.30.1 |
| cudf | 26.10.00a294 nightly |
| cugraph | 26.10.00a30 nightly |
| cuml | 26.10.00a76 nightly |
| pylibcugraph | 26.10.00a30 nightly |
| rmm | 26.10.00a32 nightly |
| CuPy | 14.2.0 |
| CUDA package version | 12.9 |

The environment solver dry run resolved 436 packages. The exact lock also passed an installation
dry run. Neither PyTorch nor libtorch resolved to a CPU build.

`RAPIDS_CUDA_VERSION=12.9` is persisted in the YAML and the active environment. This variable is
required by the RAPIDS dependency matrix used when building `cugraph-pyg`.

## GPU verification

An initial sandboxed check reported that `nvidia-smi` could not communicate with the driver. This
was sandbox device isolation, not a host problem. An unrestricted check found:

- NVIDIA driver 535.161.08;
- eight Tesla V100-SXM2-32GB GPUs;
- PyTorch compiled for CUDA 12.9;
- `torch.cuda.is_available() == True`;
- `torch.cuda.device_count() == 8`.

An actual CUDA allocation and kernel completed successfully:

```text
device Tesla V100-SXM2-32GB
result [0, 1, 4, 9, 16, 25, 36, 49]
```

## Build

The requested build was performed with the isolated environment:

```bash
./build.sh libwholegraph pylibwholegraph cugraph-pyg tests --enable-tiledb
```

Two setup issues were diagnosed during the build:

1. activating `tiledb-wg` by name failed because Conda did not discover mamba's custom environment
   directory, so subsequent commands activated its absolute prefix;
2. the repository's existing CMake cache referenced compilers and RAFT in the old `rapids` prefix.
   Repository build artifacts were cleaned and the project was configured again from a fresh cache.

The fresh CMake cache confirmed:

```text
BUILD_WITH_TILEDB=ON
CMAKE_CUDA_COMPILER=/raid/abarghi/.local/share/mamba/envs/tiledb-wg/bin/nvcc
CMAKE_CXX_COMPILER=/raid/abarghi/.local/share/mamba/envs/tiledb-wg/bin/x86_64-conda-linux-gnu-c++
CMAKE_INSTALL_PREFIX=/raid/abarghi/.local/share/mamba/envs/tiledb-wg
CUDA architecture=70-real
CUDA compiler=12.9.86
```

The C++ library, TileDB ingest tool, tests, benchmark target, and `pylibwholegraph` built and
installed successfully. The first `cugraph-pyg` packaging attempt failed because
`RAPIDS_CUDA_VERSION` was empty. Rebuilding that target with `RAPIDS_CUDA_VERSION=12.9` succeeded,
and the variable was then persisted in the environment specification.

Installed source-built package versions were:

```text
pylibwholegraph 26.10.00
cugraph-pyg     26.10.00
```

## C++ TileDB storage tests

Command:

```bash
./cpp/build/gtests/TILEDB_STORAGE_TEST --gtest_color=no
```

Initial run before the Cython error-handler change:

```text
2 tests passed; 155 ms total
```

Fresh run after the handler change:

| Test | Result | Time |
|---|---:|---:|
| `TileDBStorage.PreservesRequestOrderDuplicatesAndColumnSlices` | PASS | 95 ms |
| `TileDBStorage.AcceptsGlobalIdsForALocalPartition` | PASS | 49 ms |
| **Total** | **2/2 PASS** | **144 ms** |

Post-fix rebuild run:

```text
2/2 passed; 682 ms total
```

## One-rank gather verification

A temporary test harness created rank-local TileDB arrays with
`wholememory_tiledb_ingest`, created an equivalent pinned-CPU distributed WholeMemory tensor, and
compared outputs with exact tolerances (`rtol=0`, `atol=0`).

Fresh post-handler result:

| Case | IDs/shape | Result |
|---|---|---:|
| Unsorted IDs, duplicates, and boundary values | IDs `[0, 11, 4, 5, 5, 1, 4]`, output `[7, 4]` | PASS |
| Empty gather | output `[0, 4]` | PASS |
| Subcolumn view | columns 1–2, output `[7, 2]` | PASS |
| One-dimensional tensor | output `[7]` | PASS |

Every TileDB result exactly matched the CPU backend. The matrix was rerun after the local-mapping
and list-partition fixes using ordinary Python lists for the TileDB partition argument; all cases
passed, and local mapping returned `WHOLEMEMORY_NOT_SUPPORTED`.

## Two-rank same-node verification

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  /tmp/tiledb_wg_verify.py --root <temporary-directory>
```

The test used uneven entry partitions `[5, 7]`, making global IDs 4 and 5 the partition boundary.

| Rank | Cross-rank/duplicate IDs | Matrix | Empty | Subcolumns | 1-D |
|---:|---|---:|---:|---:|---:|
| 0 | `[0, 11, 4, 5, 5, 1, 4]` | PASS | PASS | PASS | PASS |
| 1 | `[11, 0, 5, 4, 5, 10, 4]` | PASS | PASS | PASS | PASS |

Both ranks exactly matched the pinned-CPU backend. The initial run, post-handler rerun, and final
post-fix rerun all passed. The final run used Python list partitions for both the 2-D and 1-D TileDB
tensors and confirmed local-mapping rejection on both ranks.

## Unsupported operations

Fresh post-handler results at both one and two ranks were:

| Operation | C++ result | Python result | Verification |
|---|---|---|---:|
| Scatter | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| File load | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| File store | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| Core pylibwholegraph local mapping | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |

Before the fix, local mapping produced a shaped, null-backed view. The handle-level local-memory
path now rejects `WHOLEMEMORY_ML_TILEDB` before exposing a pointer. This passed the focused Python
regression test and the one-rank and two-rank verification matrices. cuGraph-PyG continues to guard
the operation with `TypeError("TileDB-backed DistTensor has no addressable local tensor")`.

The regression test is
`python/pylibwholegraph/pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py`. It verifies a
list partition, exact gather values, invalid partition validation, and local-mapping rejection.

```bash
TEST_WM_TILEDB=1 pytest -q \
  python/pylibwholegraph/pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py
```

Result:

```text
1 passed in 4.23s
```

## Error code 9 investigation and change

`WHOLEMEMORY_NOT_SUPPORTED` is enum value 9. Scatter, load, and store were returning the correct
C++ status. The error was in Cython's `check_wholememory_error_code()`: although its public enum
exposed `NotSupported`, the translator had no corresponding branch and fell through to
`Error code 9 not recognized`.

The following handler was added to
`python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx`:

```cython
elif err_code == NotSupported:
    raise NotImplementedError('Not supported')
```

`pylibwholegraph` was rebuilt and reinstalled. Runtime tests then confirmed that scatter, file load,
and file store produce the translated message. Gather correctness was unchanged.

The local-mapping failure was separate. The Python DLPack path calls the handle-level
`wholememory_get_local_memory()` function rather than the tensor-map API that already rejected
TileDB. `get_local_memory_from_handle()` now detects `WHOLEMEMORY_ML_TILEDB` and returns
`WHOLEMEMORY_NOT_SUPPORTED` before calling the base virtual method.

## cuGraph-PyG verification

A `DistTensor` was opened with `backend="tiledb"` and a rank-template URI. The fresh post-handler
smoke test verified:

- duplicate/unsorted gather values against a CPU tensor;
- assignment rejection for the read-only tensor;
- local-view rejection at the cuGraph-PyG layer.

Result:

```text
cugraph-pyg TileDB DistTensor gather/read-only/local-view checks passed
```

## Benchmark

The benchmark used a one-rank 131,072 × 64 float32 tensor (32 MiB), fixed-seed identical traces,
tile extents 64/1,024/8,192, and batch sizes 128/2,048/16,384. The locality trace selected 90% of
IDs from 16 clusters with 256-row windows and 10% uniformly at random. Each TileDB configuration
was spot-checked against the CPU backend before timing.

Warm-cache p50/p95 latency in milliseconds:

| Trace | Batch | Pinned CPU | Tile 64 | Tile 1,024 | Tile 8,192 |
|---|---:|---:|---:|---:|---:|
| Random | 128 | 0.347/0.397 | 23.753/26.228 | 27.125/30.471 | 29.098/30.556 |
| Random | 2,048 | 0.379/0.422 | 43.844/46.769 | 38.540/39.355 | 36.790/38.469 |
| Random | 16,384 | 0.811/0.854 | 79.370/81.507 | 60.064/61.365 | 59.563/65.239 |
| Locality | 128 | 0.343/0.388 | 23.836/27.056 | 23.224/27.223 | 28.421/32.290 |
| Locality | 2,048 | 0.370/0.403 | 36.211/39.755 | 39.480/45.489 | 38.450/42.446 |
| Locality | 16,384 | 0.604/0.619 | 44.149/46.392 | 40.415/40.702 | 36.834/39.266 |

Peak CUDA allocations by batch size were:

| Batch | Peak CUDA allocation |
|---:|---:|
| 128 | 2.82 MB |
| 2,048 | 4.33 MB |
| 16,384 | 15.57 MB |

The benchmark completed in 21.29 seconds and reached 1,131,616 KiB maximum RSS. It was not a cold
NVMe benchmark: `/tmp` is ext4 on rotational `/dev/sda`, and `/proc/self/io` recorded zero physical
reads after warmup. These numbers therefore characterize warm-cache TileDB query, CPU, routing, and
staging overhead. The benchmark was not rerun after the Cython-only error-translation change because
that change does not touch the gather path.

## DGX Spark GB10 cold-NVMe benchmark

A later run on host `p4242-0660` used one NVIDIA GB10, `/dev/nvme0n1p2` with ext4, and an 8 GiB
synthetic feature table containing 16,777,216 rows of 128 float32 values. The checked-in raw results
and report are under `nvme_benchmark_results/20260821-gb10/`.

The results confirmed two different performance limitations:

- for a 65,536-row locality-biased gather, the best cold TileDB configuration used 256-row tiles,
  reached 0.525 GiB/s useful throughput, and had 1.27x read amplification; pinned CPU reached
  30.462 GiB/s;
- cold random gathers had 253x to 16,069x read amplification, while warm random gathers remained
  hundreds to thousands of milliseconds with zero physical reads, demonstrating substantial
  TileDB query/range CPU overhead in addition to storage amplification.

The two complete GB10 runs were directionally repeatable: their median mean-latency difference was
approximately 2.2%, and their cold physical-read counts matched. The original ten samples per case
only provide a directional p95.

The benchmark has since been extended for the RTX 6000 Pro run. It now supports one process per GPU,
rank-local TileDB arrays, cross-rank global IDs, synchronized cache control, slowest-rank latency,
block-device and per-process I/O counters, raw per-sample output, optional TileDB statistics,
CUDA allocator and RSS observations, TileDB staging-allocation/D2H/read-and-reorder/H2D timings,
recorded sampler traces, consolidated-array experiments, and bounded query-chunk experiments. The
default backend behavior and default TileDB tile extent remain unchanged until these experiments
provide representative evidence.

The planned eight-GPU command is:

```bash
python python/pylibwholegraph/benchmarks/tiledb_feature_fetch_benchmark.py \
  --world-size 8 \
  --data-dir /path/on/nvme/wholememory-tiledb-rtx6000 \
  --output nvme_benchmark_results/rtx6000-pro-8gpu.json \
  --tile-extents 16,64,256,4096,65536 \
  --query-chunk-rows 0,1024,4096,16384 \
  --repetitions 30 \
  --storage-baseline \
  --tiledb-stats
```

Before the benchmark, run the checked-in correctness regression on all eight GPUs:

```bash
TEST_WM_TILEDB=1 TEST_WM_TILEDB_WORLD_SIZE=8 pytest -q \
  python/pylibwholegraph/pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py
```

Run a second pass with `--consolidate` and a different output/data directory to compare fragment
layout without overwriting the unconsolidated arrays. Device-level read counters can include other
host activity, so the RTX host should be otherwise idle during measurement.

## Eight-GPU RTX PRO 6000 NVMe verification

The planned run completed on host `4u8g-tur-0037` at commit
`0ecd664a2f29029f3fbadc3b3d5b7d569767cd13`. The host provided:

- eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 97,887 MiB each;
- NVIDIA driver 595.71.05;
- local ext4 storage on `/dev/nvme1n1p1` mounted at `/raid`, backed by a 2.9 TB KIOXIA
  KCD81VUG3T20 NVMe device;
- PyTorch 2.10.0 built for CUDA 12.9, with all eight GPUs visible and usable.

The exact locked environment was recreated from `conda/environments/tiledb-wg-linux-64.lock` and
the TileDB-enabled library, Python packages, cuGraph-PyG package, and tests were built for native
`120a-real` architecture. The C++ TileDB storage suite passed 3/3 tests in 550 ms. The focused
Python regression passed with eight ranks, including exact gather results, list-partition handling,
validation, and local-mapping rejection on every rank.

The literal pytest command in this document exposed a launch-environment issue in the fresh wheel
build: multiprocessing children imported the source-tree package, which does not contain the
installed compiled extension. Invoking the unchanged checked-in test function as an importable
module with the installed wheel first on `PYTHONPATH` ran the same eight-process test body and
passed. This was a Python import-shadowing failure before GPU initialization, not a backend test
failure.

Both full benchmark passes used 16,777,216 rows by 128 float32 columns (8 GiB), all five requested
tile extents, all four query-chunk settings, 30 repetitions, the storage baseline, and TileDB
statistics. Each pass produced 252 aggregate cases and 7,560 synchronized aggregate samples; each
of the eight rank files contains 252 cases with 30 samples apiece. All recorded mean and p95
latencies are finite.

| Pass | Cases/samples | Elapsed | Sequential NVMe baseline |
|---|---:|---:|---:|
| Unconsolidated | 252 / 7,560 | 6,212 s | 3.218 GiB/s |
| Consolidated | 252 / 7,560 | 6,266 s | 3.233 GiB/s |

Representative best unconsolidated results for the largest 65,536-row batch were:

| Cache/trace | Best tile | Query chunk | Mean/p95 latency | Useful throughput | Read amplification |
|---|---:|---:|---:|---:|---:|
| Cold random | 65,536 | 0 | 1,641.4/1,661.5 ms | 0.152 GiB/s | 32.01x |
| Cold locality | 256 | 0 | 160.4/174.2 ms | 1.558 GiB/s | 2.02x |
| Warm random | 65,536 | 0 | 376.5/390.0 ms | 0.664 GiB/s | 0x physical reads |
| Warm locality | 4,096 | 0 | 87.4/94.6 ms | 2.860 GiB/s | 0x physical reads |

Consolidation had negligible overall effect because the ingest already produced a simple fragment
layout: median latency changed by -0.38% for cold cases and -1.64% for warm cases, and median cold
physical bytes were unchanged. The best consolidated 65,536-row locality cases reached 1.622 GiB/s
cold and 2.912 GiB/s warm. The results reinforce the GB10 findings: locality and small tiles greatly
reduce cold read amplification, while warm random performance remains dominated by TileDB
query/range and CPU reorder overhead rather than storage.

Aggregate JSON, CSV, raw-sample CSV, and per-rank checkpoints are under
`nvme_benchmark_results/` with the prefixes `rtx6000-pro-8gpu` and
`rtx6000-pro-8gpu-consolidated`. The generated arrays remain outside the checkout in separate
49 GiB directories under `/raid/abarghi/wholememory-tiledb-rtx6000-{unconsolidated,consolidated}`.

## Additional findings

### Partition-list API mismatch

`create_wholememory_tensor_from_tiledb()` originally documented `tensor_entry_partition` as a
Python list but passed it directly to a typed Cython memoryview. A list raised:

```text
TypeError: a bytes-like object is required, not 'list'
```

The direct constructor now checks partition length, positivity, and total size, then converts the
partition with `numpy.asarray(..., dtype=numpy.uintp)`. Ordinary Python lists passed the focused
regression test and the final one-rank and two-rank matrices.

### Non-fatal V100 warning

WholeGraph emitted the following warning during GPU initialization:

```text
cuDeviceGetAttribute(... CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ...) failed with invalid argument
```

It was non-fatal on this driver/V100 combination and did not affect CUDA kernels, gathers, or NCCL
communication.

## Remaining work

1. [x] Reject TileDB in `get_local_memory_from_handle()` with `WHOLEMEMORY_NOT_SUPPORTED` and add a
   Python regression test for local mapping.
2. [x] Validate and convert ordinary Python lists in the direct pylibwholegraph TileDB partition
   API.
3. [ ] Run the distributed matrix across at least two physical nodes. Blocked on infrastructure:
   this session exposes only `dgx19`, no scheduler or MPI launcher, and no second node endpoint.
4. [x] Run a single-GPU cold-storage benchmark on DGX Spark GB10 NVMe with physical-byte and
   bandwidth accounting.
5. [x] Run the enhanced benchmark with eight RTX PRO 6000 ranks on local NVMe, including TileDB
   statistics, query-chunk and consolidation sweeps, device staging observations, and an otherwise
   idle block device.
6. [ ] Capture a representative GNN sampler trace and use the built-in phase counters plus Nsight
   Systems to separate routing wait from D2H and TileDB query from CPU reorder, and to measure NCCL.

## Final status

| Area | Status |
|---|---:|
| Isolated CUDA/RAPIDS/TileDB environment | PASS |
| TileDB-enabled build | PASS |
| C++ TileDB storage tests | PASS |
| One-rank CPU comparison | PASS |
| Two-rank same-node gathers | PASS |
| Scatter/load/store rejection | PASS |
| Cython `NotSupported` translation | PASS |
| Python list partition handling | PASS |
| Focused Python TileDB regression | PASS |
| cuGraph-PyG TileDB smoke test | PASS |
| Core pylibwholegraph local mapping rejection | PASS |
| True multi-node test | BLOCKED: no second node |
| Single-GPU cold NVMe benchmark | PASS: DGX Spark GB10 |
| Eight-GPU NVMe benchmark | PASS: RTX PRO 6000, unconsolidated and consolidated |
