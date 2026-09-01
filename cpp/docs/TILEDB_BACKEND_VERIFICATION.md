# TileDB WholeGraph backend verification

Date: 2026-08-21
Branch: `prototype/tiledb-backend`
Host: `dgx19`

Updated: 2026-08-25 with the eight-GPU RTX PRO 6000 NVMe results.

Updated: 2026-08-27 with the four-rank colocated loading-benchmark runbook (focused GPU-compaction
matrix) on GPUs 4-7 / NUMA node 1.

Updated: 2026-08-28 with the full TabICLv2-oriented overlap benchmark runbook
(`TILEDB_LOADING_BENCHMARK_RUNBOOK.md`), steps 1-6: environment/build, correctness, smoke, the
focused overlap matrix, and the optional complete legacy matrix (widths 128/512/2,048), all on
GPUs 4-7 / NUMA node 1.

Updated: 2026-08-28 with the follow-on paired-overlap and multi-run runbook update (commit
`2885b66`): the corrected scattered 25%-unique pair (identical node-wide unique ID set) and the new
10/40/100-run clustered multi-run cases, on GPUs 4-7 / NUMA node 1. The unchanged optional complete
matrix was not re-run.

Updated: 2026-09-01 with `TILEDB_DIRECT_IO_IOPS_RUNBOOK.md` (commit `0558484`): rebuilt with the
new `libwholememory_tiledb_direct_io_preload.so`, reran the ICL-shaped overlap curves with
block-device IOPS counters, and ran the experimental `O_DIRECT` TileDB read path comparison
(primary 40-run case plus the optional 10/40/100-run sensitivity sweep), on GPUs 4-7 / NUMA
node 1.

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

After the RTX 6000 Pro run, the benchmark was extended with a nested `cpu_reorder_ms` timer around
the final host-side scatter that restores request order, expands duplicates, and applies column
slices. Existing GB10 and RTX 6000 Pro result files predate that timer and therefore do not contain
measured reorder values; a rerun is required. `tiledb_read_ms` remains the enclosing measurement and
also includes ID sorting/deduplication, range construction, query execution, and the scatter.

### Post-run GPU compaction change

The implementation now sorts and deduplicates owner-local IDs on GPU after WholeMemory routing.
Only sorted unique IDs are copied to the CPU, TileDB returns only unique rows, and only those rows
are copied H2D. WholeMemory's CUDA gather kernel then expands them into the existing NCCL send-buffer
order. New timers report `gpu_sort_ms`, `gpu_deduplicate_ms`, and `gpu_expand_ms`; the existing ID D2H
and row H2D timers now measure compact transfers. CPU sort/deduplication are bypassed, and full-row
gathers no longer need the CPU reorder/copy loop. The RTX results above predate this change and must
not be used to estimate its speedup; rerun correctness and the loading matrix before drawing a new
performance conclusion.

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

### TileDB range and tile-planning time

The backend first sorts and deduplicates rank-local IDs, coalesces adjacent IDs into inclusive
ranges, and adds those ranges to a TileDB subarray. TileDB then identifies the fragments and dense
tiles overlapping the ranges, partitions the query to fit its internal memory budget, schedules VFS
reads, unfilters tiles, and copies selected cells into the result buffer. Random IDs generate nearly
one range per unique row and can touch every large tile; locality produces fewer ranges, while tile
extent trades tile-count/metadata overhead against read amplification.

`--tiledb-stats` measured the internal planning work on sample 0 of every RTX case. The following
values are the maximum rank timer in a representative 65,536-row unconsolidated case, converted to
milliseconds:

| Cache/trace | Tile | Chunk | End-to-end | Tile-overlap planning | Next-partition | Tile reads | Reader work |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold random | 65,536 | 0 | 1,628.4 | 48.2 | 63.6 | 1,392.4 | 1,544.4 |
| Warm random | 65,536 | 0 | 362.8 | 53.1 | 67.9 | 161.6 | 297.2 |
| Cold locality | 256 | 0 | 161.7 | 34.2 | 40.5 | 90.9 | 137.5 |
| Warm locality | 4,096 | 0 | 86.7 | 32.4 | 39.4 | 14.2 | 58.1 |
| Warm locality | 4,096 | 1,024 | 1,223.4 | 824.5 | 919.4 | 46.4 | 1,139.9 |

These TileDB timers are nested and are not additive wall-clock phases. In particular,
`read_compute_relevant_tile_overlap` closely overlaps `read_compute_tile_overlap`, and reader work
contains planning, reads, unfiltering, and copying. They nevertheless show that query chunking
repeats planning and can make it dominant: the 1,024-row chunk increased overlap planning from
32.4 ms to 824.5 ms while warm tile-read time only increased from 14.2 ms to 46.4 ms.

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

## Four-rank colocated loading benchmark (RTX PRO 6000 GPUs 4-7, NUMA node 1)

Executed the operational runbook in
[`TILEDB_LOADING_BENCHMARK_RUNBOOK.md`](TILEDB_LOADING_BENCHMARK_RUNBOOK.md) on host
`4u8g-tur-0037` at commit `801e67dba63cecbc10ff0a9ee117164d2269fb9a` (tip of
`prototype/tiledb-backend`; working tree clean, no local remote named `fork` — `origin` already
points at the fork).

- Data/result paths: `/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z` and
  `/raid/abarghi/wholememory-tiledb-loading-results-20260827T194011Z` (kept on `/raid`, not in
  Git).
- Environment: recreated activation of `/raid/abarghi/.local/share/mamba/envs/tiledb-wg`
  (`RAPIDS_CUDA_VERSION=12.9`). `./build.sh libwholegraph pylibwholegraph tests --enable-tiledb`
  succeeded.
- Preflight: GPUs 4-7 idle/healthy (0% util, 14 MiB used each), NUMA affinity 1 for both GPUs 4-7
  and the `nvme1` controller, 1.5 TiB free on `/raid`, 2.2 TiB available memory, no unrelated
  `nvme1n1` traffic.
- C++ TileDB storage tests: 5/5 PASS.
- Correctness regression (`TEST_WM_TILEDB_WORLD_SIZE=4`, `CUDA_VISIBLE_DEVICES=4,5,6,7`,
  `numactl --cpunodebind=1 --membind=1`): 1 passed.
- Smoke matrix (`ROWS=262144 WIDTHS=128 REPETITIONS=2 WARMUP=1`): both CSVs, the aggregate JSON,
  and 4 rank checkpoints present and nonempty; CPU, rank-local TileDB, and node-shared TileDB rows
  all present; all latencies finite/positive; GPU sort/dedup/expand present with CPU sort/dedup at
  zero; 72 cold TileDB rows all reported nonzero `device_read_gib`; every rank's CPU affinity fell
  in NUMA node 1 (cores 64-127/192-255).
- Focused GPU-compaction matrix (`run_tiledb_gpu_compaction_benchmark.sh`, new dataset at
  `.../wholememory-tiledb-loading-20260827T194011Z/full`): completed with exit 0 in
  approximately 22.5 minutes total (19:47:28-20:10:00 UTC). Per-phase elapsed: width 128 locality
  ~35 s, width 512 locality ~10 m 21 s, width 2,048 locality ~7 m 16 s, random sentinel ~1 m 55 s,
  layout spot check ~2 m 25 s.

| Result set | Cases | Samples |
|---|---:|---:|
| `gpu-compaction-locality-width-128` | 20 | 100 |
| `gpu-compaction-locality-width-512` | 20 | 100 |
| `gpu-compaction-locality-width-2048` | 20 | 100 |
| `gpu-compaction-random-width-2048` | 3 | 9 |
| `gpu-compaction-layout-width-2048` | 8 | 40 |
| **Total** | **71** | **349** |

All 349 samples matched the expected count from the runbook. Every full-row locality case reported
`id_sort_mean_ms == 0`, `id_deduplicate_mean_ms == 0`, and `cpu_reorder_mean_ms == 0` (CPU
sort/dedup/reorder bypassed by GPU compaction, as expected). No NaN/infinite/non-positive latencies
and no `storage_unique_rows_mean > storage_requested_rows_mean` violations were found across any
result set, so the optional complete matrix (runbook step 6) was not run.

Sequential cold-storage baseline (buffered sequential read of all rank partitions):

| Width | Total bytes | Mean throughput |
|---:|---:|---:|
| 128 | 8 GiB | 3.268 GiB/s |
| 512 | 32 GiB | 3.173 GiB/s |
| 2,048 | 128 GiB | 3.200 GiB/s |

### Deviations from the runbook

1. `git fetch fork` — no `fork` remote exists in this checkout; `origin` already points at the
   fork (`alexbarghi-nv/cugraph-gnn`), so `git fetch origin` was used instead. The branch was
   already at tip.
2. Activating the environment via `micromamba activate` in a non-interactive shell did not run
   the env's `etc/conda/activate.d` hooks, so `LD_LIBRARY_PATH` was not set and
   `import pylibwholegraph.binding.wholememory_binding` failed with a missing-`.so` error.
   Explicitly exporting `LD_LIBRARY_PATH=<env>/lib` fixed this.
3. The runbook's literal correctness command
   (`python -m pytest ... pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py`) reproduces
   the same import-shadowing failure mode already described above under "Eight-GPU RTX PRO 6000
   NVMe verification," but this time in the four spawned rank subprocesses rather than the parent:
   pytest's package-name resolution walks the `__init__.py` chain and re-registers
   `sys.modules['pylibwholegraph']` from the source tree (no compiled extension), and that
   source-tree resolution is what each `multiprocessing` "spawn" rank re-imports by dotted name,
   since the installed wheel does not ship a `tests` subpackage for that dotted path to resolve
   against. Fixed without altering the test by (a) pre-importing the installed
   `pylibwholegraph`/`pylibwholegraph.binding.wholememory_binding` before invoking `pytest.main()`,
   and (b) symlinking the source `pylibwholegraph/tests` directory into the installed
   site-packages `pylibwholegraph/` so the dotted path
   `pylibwholegraph.tests.pylibwholegraph.test_tiledb_tensor` resolves identically, and
   consistently against the compiled binding, in both the parent and every spawned rank process.
   Invoked as `pytest -q --pyargs pylibwholegraph.tests.pylibwholegraph.test_tiledb_tensor`.

## TabICLv2-oriented overlap benchmark + optional complete matrix (2026-08-28)

- Host: `4u8g-tur-0037`. Branch commit: `1b01c883c8953a6659635a6b9811b995ceb5bc1e`
  (`prototype/tiledb-backend`, matches `origin`).
- Environment/CUDA versions: Python 3.12.13; PyTorch 2.10.0/CUDA 12.9; `RAPIDS_CUDA_VERSION=12.9`;
  driver 595.71.05 / CUDA 13.2 (see `environment.txt` in the result directory for the full
  `print_env.sh` capture).
- Data dir: `/raid/abarghi/wholememory-tiledb-tabicl-20260827T230559Z` (smoke only; the focused and
  full matrices reused an existing same-day data directory — see deviation 4 below).
- Result dir: `/raid/abarghi/wholememory-tiledb-tabicl-results-20260827T230559Z`.
- Reused full data dir: `/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z/full`.

### Pass/fail status and elapsed time

| Step | Status | UTC window | Elapsed |
|---|---|---|---:|
| Build (`libwholegraph`, `pylibwholegraph`, tests, `--enable-tiledb`) | PASS | — | — |
| C++ `TILEDB_STORAGE_TEST` | PASS: 5/5 | — | — |
| Correctness regression (`TEST_WM_TILEDB_WORLD_SIZE=4`) | PASS: 1/1 | 23:15:52-23:15:58 | ~6 s |
| Smoke matrix | PASS (automated checks) | 23:15:58-23:16:04 | ~6 s |
| Focused TabICLv2 overlap matrix | PASS | 23:16:04-23:24:50 | ~8 m 46 s |
| Optional complete matrix, width=128 | PASS | 23:24:50-23:33:19 | ~8 m 29 s |
| Optional complete matrix, width=512 | PASS | 23:33:19-00:01:24 | ~28 m 5 s |
| Optional complete matrix, width=2,048 | PASS | 00:01:24-01:38:57 | ~1 h 37 m 33 s |
| **Total (correctness through full matrix)** | | 23:15:52-01:38:57 | **~2 h 23 m** |

### Aggregate case/sample counts

| Result set | Aggregate cases | Samples | Expected |
|---|---:|---:|---|
| Focused clustered (`tabicl-overlap-clustered-width-2048`) | 42 | 210 | 42/210 |
| Focused scattered (`tabicl-overlap-scattered-width-2048`) | 6 | 30 | 6/30 |
| Focused continuity (`tabicl-continuity-width-2048`) | 6 | 30 | 6/30 |
| Full matrix, width=128 (`loading-width-128`) | 156 | 1,560 | 156/1,560 |
| Full matrix, width=512 (`loading-width-512`) | 156 | 1,560 | 156/1,560 |
| Full matrix, width=2,048 (`loading-width-2048`) | 156 | 1,560 | 156/1,560 |

All six result sets matched their expected counts exactly, and no `NaN`/infinite/non-positive
`latency_mean_ms`/`latency_p50_ms`/`latency_p95_ms` were found in any of them.

### Invariant and affinity checks

All seven overlap cases in the clustered result matched the runbook's invariant table exactly
(within-rank/node-wide unique fraction, repetition, requesting ranks per unique row):
`independent` 100/100/1x/1, `cross_rank_25` 100/25/1x/4, `within_rank_25` 25/25/4x/1,
`combined_12_5` 50/12.5/2x/4, `combined_6_25` 25/6.25/4x/4, `combined_3_125` 12.5/3.125/8x/4,
`stress_1` 4/1/25x/4.

Every CPU-backend aggregate row across all six result sets reported
`gpu_sort_mean_ms == 0`/`gpu_deduplicate_mean_ms == 0` (no GPU-path contamination). All 238 cold
TileDB aggregate rows across the run reported nonzero `storage_read_gib` (physical reads from
`/dev/nvme1n1`, not just a `POSIX_FADV_DONTNEED` hint). All 28 rank checkpoints reported
`cpu_affinity` entirely within NUMA node 1's core set (64-127, 192-255); `CUDA_VISIBLE_DEVICES=4,5,6,7`
was passed explicitly in every invocation (rank checkpoints record the GPU model but not the
physical index, so this was verified by construction rather than from the JSON).

Disk headroom on `/raid` held throughout: 843 GiB free before the run, 683 GiB before the
width=2,048 stage, 298 GiB free at completion (2.5 TiB used of 2.9 TiB, 90%).

### Anomaly

12 of 28 rank checkpoints (the width=128/512/2,048 full-matrix ranks, which pass `--tiledb-stats`)
recorded `"tiledb_stats_available": true` with
`"tiledb_stats_error": "...: undefined symbol: tiledb_stats_enable"`. The native TileDB internal
statistics API failed to link/resolve in this environment. This does not affect the measured
WholeMemory-level stage timers (`gpu_sort_mean_ms`, `internal_tile_read_mean_ms`,
`internal_reader_work_mean_ms`, etc., all populated and finite in every checked row) — only the
lower-level native TileDB planning/tile-read counters that `--tiledb-stats` additionally requests
are unavailable. Smoke and the focused overlap matrix do not pass `--tiledb-stats` and were
unaffected.

### Deviations from the runbook

4. Reused the existing data directory `/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z/full`
   (from a same-day run, not created via this runbook's own naming convention) as the focused-matrix
   `focused_data_dir` and the full-matrix data directory, instead of provisioning a fresh one. Its
   `.wholememory_benchmark.json` markers matched 16,777,216 rows, width 2,048, tile extent 256, node
   layout. This was necessary because `/raid` had only 843 GiB free at preflight time (below the
   1.3 TiB "from scratch" budget for the optional complete matrix); reusing the 633 GiB already
   provisioned there made the disk budget work. No existing dataset or result directory was deleted
   or overwritten.
5. `micromamba info --base` did not resolve a usable profile-script path in this shell
   (`.../etc/profile.d/micromamba.sh: No such file or directory`); used
   `eval "$(micromamba shell hook --shell bash)"` instead, which is functionally equivalent.
6. `cpp/build/gtests/...`/build steps aside, running `python -m pytest` (default `--import-mode`)
   against `test_tiledb_tensor.py` at `TEST_WM_TILEDB_WORLD_SIZE=4` still shadowed the installed
   binding in the four spawned rank subprocesses even after fixing `LD_LIBRARY_PATH` (deviation 2
   from the prior update), because `multiprocessing`'s `spawn` start method inherits the *parent's*
   live `sys.path` at `Process.start()` time, and pytest's default "prepend" rootpath resolution had
   already put the unbuilt source tree (`python/pylibwholegraph`) ahead of site-packages in that
   `sys.path` due to the `tests/pylibwholegraph/` directory name colliding with the `pylibwholegraph`
   package one level up. `--import-mode=importlib` made this worse (it shadowed even the parent
   process). Fixed, without altering the test, by adding `--import-mode=append`, which appends
   pytest's rootpath instead of prepending it, so the installed wheel (already earlier in
   `sys.path` from normal interpreter startup) resolves first in both the parent and every spawned
   rank. Invoked as:
   `pytest -q --import-mode=append python/pylibwholegraph/pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py`.
7. `print_env.sh` is not marked executable in this checkout (`-rw-r--r--`); ran it as
   `bash print_env.sh` instead of `./print_env.sh`.

## Follow-on paired-overlap and multi-run benchmark (2026-08-28)

- Host: `4u8g-tur-0037`. Branch commit: `2885b66a1e7181072caac6458b69f55030afbad5`
  (`prototype/tiledb-backend`, "Add paired overlap and multi-run TileDB benchmarks").
- No rebuild was needed: the update only touched documentation and Python benchmark/validator
  scripts, not `cpp/` sources.
- Data dir: `/raid/abarghi/wholememory-tiledb-tabicl-20260828T191645Z` (smoke only).
- Result dir: `/raid/abarghi/wholememory-tiledb-tabicl-results-20260828T191645Z`.
- Reused full data dir: `/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z/full` (same
  dataset as the 2026-08-28 TabICLv2 overlap run above; feature values and TileDB schema are
  unchanged by this update).
- Scope: steps 1-5 plus the checked-in validator only. Step 6 (optional complete matrix) was not
  re-run — nothing in that step changed, it already passed on 2026-08-28 above, and the runbook
  says to only repeat it when a focused result reveals a regression. Confirmed with the requester
  before proceeding.

### Pass/fail status and elapsed time

| Step | Status | UTC window | Elapsed |
|---|---|---|---:|
| Correctness regression (`TEST_WM_TILEDB_WORLD_SIZE=4`) | PASS: 1/1 | 19:17:25-19:17:31 | ~6 s |
| Smoke matrix (corrected scattered pair + 10/100-run cases) | PASS (automated checks) | 19:17:31-19:17:38 | ~7 s |
| Focused overlap matrix (clustered/scattered/multi-run/continuity) | PASS | 19:17:38-20:23:44 | ~1 h 6 m 6 s |
| Checked-in validator | PASS | 20:23:44-20:23:45 | ~1 s |
| **Total** | | 19:17:25-20:23:45 | **~1 h 6 m 20 s** |

### Aggregate case/sample counts

| Result set | Aggregate cases | Samples | Expected |
|---|---:|---:|---|
| Smoke (`tabicl-overlap-smoke`) | 24 | 48 | 24/48 |
| Focused clustered (`tabicl-overlap-clustered-width-2048`) | 42 | 210 | 42/210 |
| Focused scattered (`tabicl-overlap-scattered-width-2048`) | 6 | 30 | 6/30 |
| Focused multi-run (`tabicl-overlap-multirun-width-2048`) | 18 | 90 | 18/90 |
| Focused continuity (`tabicl-continuity-width-2048`) | 6 | 30 | 6/30 |
| **Focused matrix total** | **72** | **360** | 72/360 |

The checked-in validator (`validate_tiledb_tabicl_overlap_results.py`) reported:

```text
PASS: 72 aggregate configurations, 360 measured samples, paired traces and rank-aware phases valid
```

### Invariant and affinity checks

The seven clustered overlap cases again matched the runbook's invariant table exactly
(`independent` 100/100/1x/1 through `stress_1` 4/1/25x/4 — identical to the 2026-08-28 TabICLv2
run above; the underlying dataset and topology generator for this case set are unchanged).

The validator's own "paired traces ... valid" check covers the new pairing requirement: for every
scattered and multi-run sample, `cross_rank_25` and `within_rank_25` must share
`node_unique_id_sha256`, `node_contiguous_ranges`, and `node_estimated_tiles_touched`, and each
multi-run placement's `node_contiguous_ranges` must equal 10/40/100 with
`owner_unique_max_to_mean <= 1.01`. A manual spot check of the multi-run samples independently
confirmed the range-count and owner-balance invariants (0 violations across all 90 samples).

Across all five result sets (smoke + 4 focused): 0 CPU-backend `gpu_sort_mean_ms`/
`gpu_deduplicate_mean_ms` violations, 0 nonfinite/non-positive latency values, and all 32 cold
TileDB aggregate rows reported nonzero `storage_read_gib`. All 20 rank checkpoints reported
`cpu_affinity` entirely within NUMA node 1 (cores 64-127, 192-255).

Disk headroom on `/raid` held at 298 GiB free throughout (unchanged from the end of the prior run,
since only the small smoke dataset was newly provisioned).

### Deviations from the runbook

Same as the 2026-08-28 TabICLv2 overlap run above (`git fetch fork`/`micromamba info --base`
substitutions, `LD_LIBRARY_PATH`, `pytest --import-mode=append`, `print_env.sh` permissions);
applied proactively this time since they were already known. No new deviations were required.

## TileDB Direct I/O and IOPS benchmark (2026-09-01)

- Host: `4u8g-tur-0037`. Branch commit: `05584840db21a1e7c85129b37390ebc42703e3f5`
  (`prototype/tiledb-backend`, "Add TileDB Direct I/O and IOPS benchmarks").
- Rebuild: `cmake --build cpp/build -j"$(nproc)"` + `cmake --install cpp/build` picked up the new
  `wholememory_tiledb_direct_io_preload` CMake target without a full reconfigure/clean rebuild;
  installed to `<env>/lib/libwholememory_tiledb_direct_io_preload.so` (22 KiB).
- Data dir (reused, unmodified): `/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z/full`.
- Result dirs: `/raid/abarghi/wholememory-tiledb-direct-io-iops-20260901T221142Z` (primary matrix)
  and `...-sensitivity` (optional 10/40/100-run sweep).
- Preflight: `/raid` confirmed backed by local `/dev/nvme1n1p1` ext4 (not network storage); logical
  block size 512 B, physical 4 KiB; GPUs 4-7 confirmed on NUMA node 1; node 1 had 148 GiB free
  (page cache reclaimed further since the 2026-08-28 runs); no unrelated GPU/storage activity.
- Scope: primary matrix (step 3, required) and the optional sensitivity sweep (step 4) — run
  because the primary Direct I/O case passed every acceptance check and produced a latency within
  noise of buffered-cold, which the runbook treats as the trigger to extend it.

### Pass/fail status and elapsed time

| Step | Status | UTC window | Elapsed |
|---|---|---|---:|
| Rebuild + install | PASS | — | — |
| Primary matrix (direct-io-multirun + 3 buffered IOPS reruns) | PASS | 22:12:01-22:21:05 | ~9 m 4 s |
| Optional sensitivity sweep (10/40/100-run) | PASS | 22:21:05-22:30:53 | ~9 m 48 s |
| **Total** | | 22:12:01-22:30:53 | **~18 m 52 s** |

### Aggregate case/sample counts

| Result set | Aggregate cases | Samples |
|---|---:|---:|
| Primary: `direct-io-multirun-width-2048` | 1 | 5 |
| Primary: `iops-overlap-clustered-width-2048` | 21 | 105 |
| Primary: `iops-overlap-scattered-width-2048` | 6 | 30 |
| Primary: `iops-overlap-multirun-width-2048` | 18 | 90 |
| **Primary total** | **46** | **230** |
| Sensitivity: `direct-io-multirun-width-2048` | 3 | 15 |
| Sensitivity: `iops-overlap-clustered-width-2048` | 21 | 105 |
| Sensitivity: `iops-overlap-scattered-width-2048` | 6 | 30 |
| Sensitivity: `iops-overlap-multirun-width-2048` | 18 | 90 |
| **Sensitivity total** | **48** | **240** |

Each launcher invocation ran the checked-in validator (`validate_tiledb_direct_io_iops_results.py`)
automatically at the end and reported, respectively:

```text
PASS: 46 aggregate configurations, 230 samples, IOPS complete, Direct I/O interception verified
PASS: 48 aggregate configurations, 240 samples, IOPS complete, Direct I/O interception verified
```

### Direct I/O acceptance checks

All 4 Direct I/O aggregate rows (1 primary + 3 sensitivity) satisfied every acceptance criterion
from the runbook: `cache_mode=direct`; positive intercepted open/read counts (e.g. 400 opens/200
reads for the primary 40-run case); zero open and read failures; returned bytes exactly equal to
requested bytes; aligned submitted bytes at or above requested bytes
(`direct_io_alignment_read_amplification` between 1.0000 and 1.0 across all four); and positive
physical block-device read operations (`storage_read_ops` 34,872-35,xxx range). No manual `strace`
diagnosis was needed.

### Direct I/O vs. buffered-cold comparison (same 40-run cross-rank topology)

| Run count | Direct I/O mean latency | Buffered-cold mean latency | Direct I/O IOPS | Buffered-cold IOPS | Direct I/O useful GiB/s | Buffered-cold useful GiB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 220.30 ms | 211.27 ms | 29,464.7 | 30,676.2 | 13.852 | 14.445 |
| 40 | 218.96 ms | 219.96 ms | 31,810.4 | 31,890.3 | 13.938 | 13.874 |
| 100 | 242.48 ms | 248.94 ms | 32,912.3 | 32,931.9 | 12.586 | 12.259 |

Direct I/O tracked buffered-cold within about 4% at every run count in this environment — neither
consistently faster nor slower, all differences within run-to-run noise for a 5-repetition sample.
This experimental path did not need to reduce latency to be worth preserving evidence for: whether
it reduces host page-cache pressure under TabICL model memory pressure is a separate question this
run did not measure and is not decided here.

### Other checks

Across all 8 aggregate result files (primary + sensitivity): 0 non-finite/non-positive latency
values, 0 CPU-backend GPU-path (`gpu_sort_mean_ms`/`gpu_deduplicate_mean_ms`) violations, and all 30
cold TileDB aggregate rows reported nonzero `storage_read_gib`. All 32 rank checkpoints reported
`cpu_affinity` entirely within NUMA node 1 (cores 64-127, 192-255). `/raid` free space held at
298 GiB throughout (no new arrays were created; this run only measured against the existing
dataset).

### Deviations from the runbook

None required. The environment/import-mode fixes from the prior TabICLv2 overlap runs were already
in place and did not need to be reapplied (this runbook does not invoke `pytest`). The literal
`cmake --build cpp/build` command picked up the CMakeLists.txt change and re-triggered CMake's
configure step automatically; no manual reconfigure was necessary.

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
6. [x] Run the new four-rank colocated loading matrix on CPU socket/NUMA node 1 and RTX PRO 6000
   GPUs 4-7. Compare pinned CPU with rank-local and node-shared TileDB arrays across vector widths,
   locality windows, and cache states. The benchmark now separates ID routing, D2H, decode, sort,
   deduplication, range construction, query setup/submit, CPU reorder, H2D, embedding exchange, and
   output reorder, with nested TileDB planning/I/O timers retained separately. The focused
   GPU-compaction matrix (2026-08-27) completed with no regressions. The optional complete matrix
   (156 aggregate cases/width, all three widths) also completed on 2026-08-28 with no regressions
   (see "TabICLv2-oriented overlap benchmark + optional complete matrix" above) — node-side evidence
   is preserved, but a final performance conclusion should still not be drawn from a single node-side
   run alone.
7. [x] Run the focused TabICLv2-oriented overlap matrix (independent/cross-rank/within-rank/combined
   overlap topologies, scattered placement, and locality-window continuity sentinels) described in
   `TILEDB_LOADING_BENCHMARK_RUNBOOK.md`. All invariants matched expectations; see above. Follow-on
   (2026-08-28): the scattered pair now shares an identical node-wide unique ID set between its two
   topologies, and 10/40/100-run clustered multi-run cases were added; all 72/72 cases and 360/360
   samples passed, including the checked-in validator's paired-fingerprint and multi-run
   range/balance checks.
8. [ ] Develop and run the end-to-end TabICLv2 fine-tuning trace-based benchmark (with real
   time-series ordering and grouped-contiguous random contexts) as a separate effort — the overlap
   matrix above deliberately does not yet model those application-specific access patterns.
9. [x] Rerun the ICL-shaped overlap curves with physical block-device IOPS counters and compare the
   primary 40-run cross-rank case with the experimental `O_DIRECT` TileDB read path
   (`TILEDB_DIRECT_IO_IOPS_RUNBOOK.md`). Direct I/O passed every acceptance check and tracked
   buffered-cold latency within ~4% across the 10/40/100-run sensitivity sweep; see above. Whether
   it reduces host page-cache pressure under real TabICL model memory pressure is still unmeasured.

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
| Four-rank colocated loading benchmark, correctness/smoke | PASS: GPUs 4-7, NUMA node 1 |
| Four-rank focused GPU-compaction matrix | PASS: 349/349 samples, no regressions |
| Focused TabICLv2-oriented overlap matrix (2026-08-28) | PASS: 54/54 cases, 270/270 samples |
| Optional complete matrix, widths 128/512/2,048 (2026-08-28) | PASS: 468/468 cases, 4,680/4,680 samples |
| Follow-on paired-overlap + multi-run matrix (2026-08-28) | PASS: 72/72 cases, 360/360 samples |
| Direct I/O + IOPS matrix, primary (2026-09-01) | PASS: 46/46 cases, 230/230 samples |
| Direct I/O + IOPS matrix, sensitivity sweep (2026-09-01) | PASS: 48/48 cases, 240/240 samples |
| Four-rank complete loading matrix (all widths, 10 reps) | NOT YET RUN |
