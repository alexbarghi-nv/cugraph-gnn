# TileDB WholeGraph backend verification

Date: 2026-08-21
Branch: `prototype/tiledb-backend`
Host: `dgx19`

## Executive summary

The TileDB-enabled library and Python packages build successfully in an isolated Conda environment.
One-rank and two-rank same-node gathers match the pinned-CPU WholeMemory backend exactly for the
tested matrix, including cross-rank IDs, duplicates, empty batches, uneven partition boundaries,
one-dimensional tensors, and subcolumn views. The C++ TileDB storage tests and the cuGraph-PyG
`DistTensor` smoke test pass.

The branch is not fully verified against the original checklist for two reasons:

1. core pylibwholegraph local mapping still succeeds unexpectedly for a TileDB tensor and produces
   a shaped, null-backed view instead of returning `WHOLEMEMORY_NOT_SUPPORTED`;
2. true multi-node execution was unavailable because this session had access to only `dgx19`, with
   no scheduler allocation or second node endpoint.

Scatter, file load, and file store correctly return `WHOLEMEMORY_NOT_SUPPORTED`. A missing Cython
handler initially displayed this as `Error code 9 not recognized`; the handler was added and these
operations now raise `NotImplementedError("Not supported")`.

### Post-verification fixes

The branch now rejects direct local-memory access to TileDB handles with
`WHOLEMEMORY_NOT_SUPPORTED`. The direct pylibwholegraph constructor also validates an explicit
partition and converts ordinary Python lists to a `numpy.uintp` array before entering Cython. These
two fixes passed CPU-side formatting, lint, Cython translation, and Python bytecode checks, but the
GPU verification matrix in this document has not yet been rerun against them.

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

Every TileDB result exactly matched the CPU backend.

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

Both ranks exactly matched the pinned-CPU backend. The initial run and the fresh post-handler rerun
both passed.

## Unsupported operations

Fresh post-handler results at both one and two ranks were:

| Operation | C++ result | Python result | Verification |
|---|---|---|---:|
| Scatter | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| File load | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| File store | `WHOLEMEMORY_NOT_SUPPORTED` (9) | `NotImplementedError("Not supported")` | PASS |
| Core pylibwholegraph local mapping | incorrectly returns success | shaped local view | **FAIL** |

Unexpected local-view results were:

- one rank: shape `[12, 4]`, offset 0;
- two-rank rank 0: shape `[5, 4]`, offset 0;
- two-rank rank 1: shape `[7, 4]`, offset 5.

The TileDB object never allocates addressable feature memory, so these are null-backed views and
must not be used. cuGraph-PyG separately guards this operation and raises
`TypeError("TileDB-backed DistTensor has no addressable local tensor")`.

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

The local-mapping failure is separate. The Python DLPack path calls the handle-level
`wholememory_get_local_memory()` function rather than the tensor-map API that already rejects
TileDB. The handle-level function calls the base virtual method and returns success unconditionally;
the TileDB subclass does not override it.

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

## Additional findings

### Partition-list API mismatch

`create_wholememory_tensor_from_tiledb()` documents `tensor_entry_partition` as a Python list but
passes it directly to a typed Cython memoryview. A list raises:

```text
TypeError: a bytes-like object is required, not 'list'
```

Using `numpy.asarray(partitions, dtype=numpy.uintp)` works. cuGraph-PyG already performs this
conversion.

### Non-fatal V100 warning

WholeGraph emitted the following warning during GPU initialization:

```text
cuDeviceGetAttribute(... CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ...) failed with invalid argument
```

It was non-fatal on this driver/V100 combination and did not affect CUDA kernels, gathers, or NCCL
communication.

## Remaining work

1. Reject TileDB in `get_local_memory_from_handle()` with `WHOLEMEMORY_NOT_SUPPORTED`, then add a
   Python regression test for local mapping.
2. Accept ordinary Python lists in the direct pylibwholegraph TileDB partition API, or validate and
   convert them before entering Cython.
3. Run the same distributed matrix across at least two physical nodes.
4. Run cold-storage benchmarks on actual NVMe while collecting physical bytes, bandwidth, TileDB
   statistics, H2D bandwidth, pinned-memory usage, and device staging usage.

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
| cuGraph-PyG TileDB smoke test | PASS |
| Core pylibwholegraph local mapping rejection | **FAIL** |
| True multi-node test | NOT RUN |
| Cold NVMe benchmark | NOT RUN |
