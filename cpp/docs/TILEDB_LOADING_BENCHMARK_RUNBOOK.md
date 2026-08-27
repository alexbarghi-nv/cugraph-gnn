# RTX PRO 6000 TileDB loading benchmark runbook

This is the operational handoff for running the single-node loading benchmark on host
`4u8g-tur-0037`. The benchmark definition and metric semantics are in
[TILEDB_LOADING_BENCHMARK.md](TILEDB_LOADING_BENCHMARK.md). Read that document before interpreting
the results.

## Scope and guardrails

- Use only RTX PRO 6000 GPUs 4-7 and CPU socket/NUMA node 1.
- Use `/dev/nvme1n1`, mounted under `/raid`, for the TileDB arrays and result files.
- Use one four-rank WholeMemory/NCCL communicator. Do not add a second communicator, training,
  sampling, or multi-node execution.
- Compare only TileDB with distributed host-pinned (`cpu`) WholeMemory in the full run.
- Do not use `sudo`, GPUs 0-3, or another user's storage.
- Do not delete or overwrite an existing dataset or result directory. Select a new directory when
  existing metadata does not match the requested run.
- Preserve the checked-in matrix after the smoke test. Record any necessary deviation before
  changing it.

The complete matrix's datasets and six TileDB copies require approximately 1,176 GiB plus metadata.
The focused GPU-compaction matrix requires approximately 632 GiB when created from scratch; reserve
at least 700 GiB for it or 1.3 TiB for the complete matrix. Width 2,048 also creates a 128 GiB
pinned-CPU tensor, so require at least 160 GiB of available system memory; 256 GiB or more is
preferred.

## 1. Check out the implementation

Use the existing clone if it is clean. The implementation begins at commit `f655f22`; run the tip
of `prototype/tiledb-backend` so that this runbook and any subsequent fixes are included.

```bash
git status --short
git fetch fork
git switch prototype/tiledb-backend
git pull --ff-only fork prototype/tiledb-backend
git log -1 --oneline --decorate
```

Stop if `git status --short` reports changes that are not yours. Record the exact output of
`git rev-parse HEAD` with the results.

## 2. Activate and verify the environment

Prefer the existing isolated environment:

```bash
source "$(micromamba info --base)/etc/profile.d/micromamba.sh"
micromamba activate /raid/abarghi/.local/share/mamba/envs/tiledb-wg
export RAPIDS_CUDA_VERSION=12.9
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import pylibwholegraph; print(pylibwholegraph.__file__)'
```

If that prefix is absent or unusable, recreate an isolated environment from
`conda/environments/tiledb-wg-linux-64.lock`; do not modify another RAPIDS environment. Confirm
that Python imports the installed package containing the compiled binding rather than an
unbuilt source-tree package.

Build the components needed by this benchmark:

```bash
./build.sh libwholegraph pylibwholegraph tests --enable-tiledb
which wholememory_tiledb_ingest
./cpp/build/gtests/TILEDB_STORAGE_TEST --gtest_color=no
```

The C++ test must pass 5/5 tests. A clean rebuild may be necessary if an old CMake cache points at a
different environment or CUDA architecture.

## 3. Preflight the node

Run these commands without `sudo` and save their output in the result directory:

```bash
hostname
date --iso-8601=seconds
nvidia-smi -L
nvidia-smi topo -m
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
lsblk -o NAME,TYPE,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS
findmnt -T /raid -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
cat /sys/class/nvme/nvme1/device/numa_node
df -h /raid
free -h
```

Confirm all of the following before proceeding:

1. GPUs 4-7 are visible, healthy, and idle.
2. `nvidia-smi topo -m` associates GPUs 4-7 with NUMA node 1.
3. `/dev/nvme1n1p1` is the partition backing `/raid`, and its `nvme1` controller reports NUMA
   node 1.
4. `/raid` has at least 700 GiB free for the focused matrix, or 1.3 TiB for the complete matrix.
5. `MemAvailable` is at least 160 GiB.
6. No unrelated workload is producing sustained traffic on `nvme1n1`.

Use `iostat -dxm 2 5 nvme1n1` when `iostat` is installed. Stop and report the mismatch instead of
silently changing the CPU, GPU, disk, or matrix selection.

Create new run directories and capture the preflight record:

```bash
run_id=$(date -u +%Y%m%dT%H%M%SZ)
data_dir=/raid/abarghi/wholememory-tiledb-loading-${run_id}
result_dir=/raid/abarghi/wholememory-tiledb-loading-results-${run_id}
mkdir -p "${data_dir}" "${result_dir}"

{
  git rev-parse HEAD
  hostname
  date --iso-8601=seconds
  nvidia-smi -L
  nvidia-smi topo -m
  numactl --hardware
  lscpu
  lsblk -o NAME,TYPE,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS
  findmnt -T /raid -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
  cat /sys/class/nvme/nvme1/device/numa_node
  df -h /raid
  free -h
} > "${result_dir}/preflight.txt" 2>&1

./print_env.sh > "${result_dir}/environment.txt" 2>&1
set -o pipefail
```

Keep the `data_dir`, `result_dir`, and `run_id` variables in the same shell for the remaining
commands.

## 4. Run correctness and smoke checks

First run the four-rank correctness regression on the selected GPUs and NUMA node:

```bash
env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  TEST_WM_TILEDB=1 \
  TEST_WM_TILEDB_WORLD_SIZE=4 \
  numactl --cpunodebind=1 --membind=1 \
  python -m pytest -q \
  python/pylibwholegraph/pylibwholegraph/tests/pylibwholegraph/test_tiledb_tensor.py \
  2>&1 | tee "${result_dir}/correctness.log"
```

All ranks must return exact values for both rank-local and node-shared arrays. If Python imports an
unbuilt source package, correct `PYTHONPATH` or run from a directory that exposes the installed
wheel first; do not alter the test to bypass the compiled binding.

Then run a small end-to-end smoke matrix:

```bash
ROWS=262144 \
WIDTHS="128" \
REPETITIONS=2 \
WARMUP=1 \
python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh \
  "${data_dir}/smoke" \
  "${result_dir}/smoke" \
  2>&1 | tee "${result_dir}/smoke.log"
```

Before the full run, confirm that:

- `smoke/loading-width-128.json`, both CSVs, and four rank checkpoints exist and are nonempty;
- CPU, rank-local TileDB, and node-shared TileDB rows are present;
- all latencies are finite and positive;
- TileDB phase metrics have `valid=true` in the rank checkpoints;
- GPU sort/dedup/expand metrics are present, CPU sort/dedup metrics are zero, and unique staging
  byte counts shrink for duplicate-heavy traces;
- cold TileDB samples report physical reads from `/dev/nvme1n1`;
- recorded CPU affinities are contained in NUMA node 1; and
- recorded `CUDA_VISIBLE_DEVICES` is `4,5,6,7`.

Do not reuse the smoke data directory for the full run because its row count differs.

## 5. Run the focused GPU-compaction matrix

Reuse the previous full-run data directory when it is still available and its
`.wholememory_benchmark.json` markers match 16,777,216 rows, the requested width, and tile extent.
The feature values and TileDB schema did not change with GPU compaction. Always select a new result
directory; never pass `--overwrite` merely to reuse valid arrays.

Run the focused launcher:

```bash
focused_data_dir=/path/to/previous/full-data-directory
# For a new dataset instead, use: focused_data_dir="${data_dir}/full"
python/pylibwholegraph/benchmarks/run_tiledb_gpu_compaction_benchmark.sh \
  "${focused_data_dir}" \
  "${result_dir}/gpu-compaction" \
  2>&1 | tee "${result_dir}/gpu-compaction.log"
```

The default focused run contains:

- 60 primary locality configurations across widths 128, 512, and 2,048;
- 3 width-2,048 random-sentinel configurations;
- 8 width-2,048 node/rank TileDB layout configurations;
- 349 measured samples and 488 synchronized rounds including warmups.

The locality matrix uses two warmups and five measured repetitions. Treat its p95 values as
directional; reserve a later 10- or 20-repetition run for configurations selected by the real
TabICLv2 trace. TileDB internal statistics are disabled in this focused pass because the preceding
run already isolated planning and tile-read behavior. The new WholeMemory stage timers remain
enabled and must show GPU sort/deduplication, compact D2H/H2D, and GPU expansion.

## 6. Optional complete matrix

Only repeat the complete matrix when a focused result reveals a regression outside the selected
locality cases. Start from new `full` subdirectories. The launcher fixes the communicator size,
GPUs, NUMA policy,
backend comparison, locality windows, cache modes, tile extents, batch sizes, and stage collection:

```bash
time python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh \
  "${data_dir}/full" \
  "${result_dir}/full" \
  2>&1 | tee "${result_dir}/full.log"
```

The run processes widths in the order 128, 512, and 2,048. It may take many hours. Do not infer a
failure merely from a long cold-random case; watch the log and device activity.

Useful monitoring commands from a second shell are:

```bash
nvidia-smi pmon -i 4,5,6,7 -s um -d 5
iostat -dxm 5 nvme1n1
watch -n 30 df -h /raid
ps -eLo pid,tid,psr,comm,args | grep tiledb_feature_fetch_benchmark
```

`iostat`, `watch`, and `nvidia-smi pmon` are observational and are not part of the measured process.
Avoid other storage-intensive activity while the benchmark is running.

### Restart behavior

Raw files and TileDB arrays with matching marker metadata are reused automatically. Per-rank JSON
files are progress checkpoints for diagnosis, but the benchmark does not resume partway through a
width. An interrupted width must be rerun from its beginning. If width 128 completed and the run
then failed during width 512, restart only the unfinished widths, for example:

```bash
WIDTHS="512 2048" \
python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh \
  "${data_dir}/full" \
  "${result_dir}/full" \
  2>&1 | tee -a "${result_dir}/full.log"
```

Do not pass `--overwrite` or remove arrays merely to restart measurement. Use a new directory if
array metadata is inconsistent.

## 7. Validate and preserve the results

The focused result directory should contain three `gpu-compaction-locality-width-W` result sets,
plus `gpu-compaction-random-width-2048` and `gpu-compaction-layout-width-2048`. Each result set has
an aggregate JSON/CSV, raw-sample CSV, and four rank checkpoints. Check for 20 aggregate cases and
100 measured samples in each default locality-width result, 3 cases and 9 samples in the random
sentinel, and 8 cases and 40 samples in the layout spot check.

For the focused results, verify that full-row locality cases report CPU sort, CPU deduplication, and
CPU reorder as zero; `index_bytes`, `raw_staging_bytes`, and `output_bytes` must scale with
`storage_unique_rows`, not `storage_requested_rows`. Preserve the stage timings even if one of the
new GPU phases is too short to produce a visibly nonzero millisecond value.

The full result directory should contain, for each width:

- `loading-width-W.json`: metadata, aggregates, raw synchronized samples, and per-rank phase data;
- `loading-width-W.csv`: aggregate comparison surface;
- `loading-width-W.samples.csv`: flattened raw samples; and
- `loading-width-W.rank-R.json`: per-rank progress/checkpoint data.

Check that all three aggregate JSON files exist, the log ended successfully, every configuration
has ten measured samples, and no latency or throughput is NaN or infinite. The default matrix
produces 156 aggregate cases and 1,560 synchronized aggregate samples per width: 12 pinned-CPU
cases and 144 TileDB cases. For cold TileDB rows, flag any zero block-device reads rather than
discarding the sample: `POSIX_FADV_DONTNEED` is a hint, and the physical counter determines whether
the sample was actually cold.

Record the following in `cpp/docs/TILEDB_BACKEND_VERIFICATION.md`:

- host, UTC run time, branch commit, environment and CUDA versions;
- exact data/result paths and total elapsed time per width;
- correctness, storage-test, smoke-test, and full-run pass/fail status;
- counts of aggregate cases and samples per width;
- whether every rank remained in NUMA node 1 and used GPUs 4-7;
- sequential storage baseline for each width;
- missing, zero, nonfinite, or anomalous measurements; and
- any deviation from this runbook.

Keep the terabyte-scale raw files and TileDB arrays under `/raid`; never add them to Git. Preserve
the complete result directory on the node. For repository handoff, add the updated verification
document, `preflight.txt`, logs, aggregate JSON/CSV files, and sample CSVs under a new dated directory
inside `nvme_benchmark_results/`. Rank checkpoint files are redundant with the aggregate JSON and
should only be committed when needed to diagnose a discrepancy.

Commit and push the result handoff to `prototype/tiledb-backend`. In the final report, give the
commit hash, `/raid` result path, elapsed time, test status, and any condition that could invalidate
the comparison. Do not draw final performance conclusions solely from the node-side run; preserve
the evidence for subsequent analysis.
