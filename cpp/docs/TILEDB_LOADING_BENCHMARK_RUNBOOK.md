# RTX PRO 6000 paired-overlap and multi-run TileDB benchmark runbook

This is the operational handoff for running the single-node loading benchmark on host
`4u8g-tur-0037`. The benchmark definition and metric semantics are in
[TILEDB_LOADING_BENCHMARK.md](TILEDB_LOADING_BENCHMARK.md). Read that document before interpreting
the results.

The newer focused IOPS rerun and `O_DIRECT` experiment have their own operational handoff:
[TILEDB_DIRECT_IO_IOPS_RUNBOOK.md](TILEDB_DIRECT_IO_IOPS_RUNBOOK.md).

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

This follow-up corrects the scattered 25%-unique topology pair so both cases use the exact same
node-wide unique ID set, records rank-aware phase aggregates, and adds 10-, 40-, and 100-run
clustered contexts distributed across WholeMemory owner partitions. The overlap matrix uses one
128 GiB raw width-2,048 dataset and one node-shared TileDB copy. When
created from scratch, reserve at least 320 GiB for data, metadata, and results. Reuse the prior
matching data directory when available. Width 2,048 also creates a 128 GiB pinned-CPU tensor, so
require at least 160 GiB of available system memory; 256 GiB or more is preferred. The optional
complete legacy matrix still requires approximately 1.3 TiB.

## 1. Check out the implementation

Use the existing clone if it is clean. Run the tip of `prototype/tiledb-backend` so that the exact
overlap generator, metrics, launcher, and this runbook are included.

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
4. `/raid` has at least 320 GiB free for a new overlap dataset, or 1.3 TiB for the optional complete
   matrix.
5. `MemAvailable` is at least 160 GiB.
6. No unrelated workload is producing sustained traffic on `nvme1n1`.

Use `iostat -dxm 2 5 nvme1n1` when `iostat` is installed. Stop and report the mismatch instead of
silently changing the CPU, GPU, disk, or matrix selection.

Create new run directories and capture the preflight record:

```bash
run_id=$(date -u +%Y%m%dT%H%M%SZ)
data_dir=/raid/abarghi/wholememory-tiledb-tabicl-${run_id}
result_dir=/raid/abarghi/wholememory-tiledb-tabicl-results-${run_id}
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

Then run a small end-to-end smoke matrix that exercises both sources of 25% node-wide uniqueness,
the corrected scattered pair, and the 10- and 100-run clustered boundaries:

```bash
env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  WHOLEMEMORY_TILEDB_COMPUTE_CONCURRENCY=8 \
  WHOLEMEMORY_TILEDB_IO_CONCURRENCY=8 \
  numactl --cpunodebind=1 --membind=1 \
  python3 python/pylibwholegraph/benchmarks/tiledb_feature_fetch_benchmark.py \
  --data-dir "${data_dir}/smoke" \
  --output "${result_dir}/smoke/tabicl-overlap-smoke.json" \
  --world-size 4 \
  --rows 262144 \
  --width 128 \
  --backends cpu,tiledb \
  --array-layouts node \
  --tile-extents 256 \
  --query-chunk-rows 0 \
  --cache-modes cold,warm \
  --overlap-cases cross_rank_25,within_rank_25 \
  --overlap-placements clustered,scattered,clustered_runs_10,clustered_runs_100 \
  --patterns overlap_clustered_cross_rank_25,overlap_clustered_within_rank_25,overlap_scattered_cross_rank_25,overlap_scattered_within_rank_25,overlap_clustered_runs_10_cross_rank_25,overlap_clustered_runs_10_within_rank_25,overlap_clustered_runs_100_cross_rank_25,overlap_clustered_runs_100_within_rank_25 \
  --batch-sizes 1000 \
  --block-device /dev/nvme1n1 \
  --warmup 1 \
  --repetitions 2 \
  2>&1 | tee "${result_dir}/smoke.log"
```

Before the full run, confirm that:

- the smoke JSON, both CSVs, and four rank checkpoints exist and are nonempty;
- 24 aggregate configurations and 48 measured samples are present;
- CPU and node-shared TileDB rows are present;
- all latencies are finite and positive;
- TileDB phase metrics have `valid=true` in the rank checkpoints;
- GPU sort/dedup/expand metrics are present, CPU sort/dedup metrics are zero, and unique staging
  byte counts shrink for duplicate-heavy traces;
- `cross_rank_25` reports within-rank uniqueness 100%, node-wide uniqueness 25%, and four
  requesting ranks per unique row;
- `within_rank_25` reports within-rank and node-wide uniqueness 25%, 4x within-rank repetition,
  and one requesting rank per unique row;
- for a fixed placement and sample, `cross_rank_25` and `within_rank_25` have identical
  `node_unique_id_sha256`, `node_contiguous_ranges`, and `node_estimated_tiles_touched`;
- `clustered_runs_10` and `clustered_runs_100` report exactly 10 and 100
  `node_contiguous_ranges`, respectively, and owner max/mean unique-row imbalance no greater than
  1.01;
- every sample records `slowest_rank`, `slowest_rank_has_storage`, `storage_owner_ranks`, and the
  rank max, rank mean, max-rank identity, storage-owner max, and storage-owner mean for every phase;
- cold TileDB samples report physical reads from `/dev/nvme1n1`;
- recorded CPU affinities are contained in NUMA node 1; and
- recorded `CUDA_VISIBLE_DEVICES` is `4,5,6,7`.

Do not reuse the smoke data directory for the full run because its row count differs.

## 5. Run the focused TabICLv2-oriented overlap matrix

Reuse the previous full-run data directory when it is still available and its
`.wholememory_benchmark.json` markers match 16,777,216 rows, the requested width, and tile extent.
The feature values and TileDB schema did not change with GPU compaction. Always select a new result
directory; never pass `--overwrite` merely to reuse valid arrays.

Run the overlap launcher. It deliberately does not yet model time-series ordering or
grouped-contiguous random contexts; those application-specific controls wait for the end-to-end
TabICLv2 trace.

```bash
focused_data_dir=/path/to/previous/full-data-directory
# For a new dataset instead, use: focused_data_dir="${data_dir}/full"
python/pylibwholegraph/benchmarks/run_tiledb_tabicl_overlap_benchmark.sh \
  "${focused_data_dir}" \
  "${result_dir}/tabicl-overlap" \
  2>&1 | tee "${result_dir}/tabicl-overlap.log"
```

The default focused run contains:

- 42 clustered configurations: seven overlap topologies at 48,000 and 100,000 requests per rank;
- 6 scattered configurations: the corrected paired 25%-unique topologies at 100,000 requests per
  rank, using an identical node-wide unique ID set for both topologies;
- 18 multi-run configurations: the paired 25%-unique topologies at 100,000 requests per rank with
  the same unique IDs arranged into 10, 40, or 100 non-overlapping runs across owner partitions;
- 6 continuity configurations: the preceding 256- and 4,096-row windows at 65,536 requests per
  rank;
- 72 aggregate configurations, 360 measured samples, and 504 synchronized rounds including
  warmups.

Every case uses width 2,048, a 256-row tile extent, one node-shared array, two warmups, and five
measurements. Treat p95 as directional; reserve a later 10- or 20-repetition run for configurations
selected by the real TabICLv2 trace. TileDB internal statistics are disabled because the preceding
run already isolated planning and tile-read behavior. WholeMemory stage timers remain enabled and
must show GPU sort/deduplication, compact D2H/H2D, and GPU expansion.

The two 25%-unique cases are the primary diagnostic pair. `cross_rank_25` has no duplicates within
a rank but sends the same unique set from all four ranks. `within_rank_25` repeats each row four
times within one rank but gives each rank a disjoint unique set. Their node-wide unique fractions
are identical. In the scattered and multi-run outputs, their exact sorted node-wide ID set must
also have the same `node_unique_id_sha256` for each synchronized sample. This holds physical
placement fixed so differences isolate routing, fan-out, NCCL exchange, and expansion rather than
different TileDB tile footprints. `stress_1` is a non-representative lower bound, not an expected
TabICLv2 workload.

The multi-run placements model a context assembled from multiple contiguous groups without yet
claiming to reproduce a TabICLv2 application trace. Runs are non-overlapping, distributed across
the four WholeMemory owner partitions, and balanced by unique-row count. Ten runs represent a
strongly grouped context, 100 runs a more fragmented context, and 40 runs an intermediate point.
Each sample records the exact node-wide range count, tile count, and unique-ID digest.

Legacy aggregate phase columns remain available for compatibility and still describe the slowest
end-to-end rank. They are not sufficient when that rank is a waiting non-owner. Each raw sample now
also records, for every timing and count field, the rank maximum, rank mean, rank that supplied the
maximum, storage-owner maximum, and storage-owner mean. Aggregate CSV rows contain the mean of
each of those rank-aware sample metrics. These components may overlap on different ranks and must
not be added to reconstruct end-to-end latency.

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

The focused result directory should contain `tabicl-overlap-clustered-width-2048`,
`tabicl-overlap-scattered-width-2048`, `tabicl-overlap-multirun-width-2048`, and
`tabicl-continuity-width-2048` result sets. Each has an aggregate JSON/CSV, raw-sample CSV, and four
rank checkpoints. Check for 42 aggregate cases and 210 measured samples in the clustered result, 6
cases and 30 samples in the scattered result, 18 cases and 90 samples in the multi-run result, and
6 cases and 30 samples in the continuity result.

Run the checked-in validator before interpreting or archiving the results:

```bash
python3 python/pylibwholegraph/benchmarks/validate_tiledb_tabicl_overlap_results.py \
  "${result_dir}/tabicl-overlap" \
  | tee "${result_dir}/tabicl-overlap-validation.txt"
```

It must report `PASS: 72 aggregate configurations, 360 measured samples`. It also verifies exact
topology-pair fingerprints and physical placement, multi-run range counts and owner balance,
finite positive latency, and the presence of every rank-aware phase field.

Verify the exact overlap invariants in every aggregate and raw sample:

| Case | Within-rank unique | Node-wide unique | Repetition | Requesting ranks/unique row |
| --- | ---: | ---: | ---: | ---: |
| `independent` | 100% | 100% | 1x | 1 |
| `cross_rank_25` | 100% | 25% | 1x | 4 |
| `within_rank_25` | 25% | 25% | 4x | 1 |
| `combined_12_5` | 50% | 12.5% | 2x | 4 |
| `combined_6_25` | 25% | 6.25% | 4x | 4 |
| `combined_3_125` | 12.5% | 3.125% | 8x | 4 |
| `stress_1` | 4% | 1% | 25x | 4 |

Compare `owner_unique_rows_max_mean` with `owner_unique_rows_mean` and retain the per-sample
`owner_unique_counts`; do not explain a slow case using uniqueness alone. Full-row cases must report
CPU sort, CPU deduplication, and CPU reorder as zero. `index_bytes`, `raw_staging_bytes`, and
`output_bytes` must scale with `storage_unique_rows`, not `storage_requested_rows`. Preserve every
stage timing, even if a short GPU phase rounds to zero milliseconds. Compare cold context
construction separately from warm/cache-resident retrieval; do not average the cache modes.

For every scattered and multi-run sample, pair rows by backend, layout, tile extent, cache mode,
batch size, placement, and sample index. The `cross_rank_25` and `within_rank_25` rows must have the
same `node_unique_id_sha256`, `node_contiguous_ranges`, and `node_estimated_tiles_touched`. Do not
accept a topology comparison when any of those fields differs. For multi-run samples,
`node_contiguous_ranges` must equal 10, 40, or 100 as named by `overlap_placement`, and
`owner_unique_max_to_mean` must be at most 1.01.

Use `*_rank_max_*` and `*_storage_owner_*` fields for phase diagnosis. The legacy unqualified phase
field may legitimately be zero when `slowest_rank_has_storage=false`; that is a waiting-rank
measurement, not evidence that no TileDB query ran. Do not sum rank maxima from different phases.

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
