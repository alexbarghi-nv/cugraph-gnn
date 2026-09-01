# TileDB Direct I/O and IOPS benchmark runbook

This is the focused RTX PRO 6000 follow-up to the corrected TabICL overlap benchmark. It reruns the
decision-relevant buffered cases with physical I/O operation counts and IOPS, then compares the
primary ICL-shaped case with an experimental `O_DIRECT` TileDB read path.

Use CPU NUMA node 1, GPUs 4-7, the node-layout array, and `/dev/nvme1n1`, matching the preceding
single-node measurements.

## What Direct I/O means here

The installed TileDB 2.30 POSIX backend normally opens local files with `O_RDONLY`. The benchmark
build now installs `libwholememory_tiledb_direct_io_preload.so`. When loaded and enabled, it:

- adds `O_DIRECT` only to read-only opens below the benchmark data directory;
- uses `statx(STATX_DIOALIGN)` when available, with a conservative 4 KiB fallback;
- rounds unaligned TileDB reads outward into aligned bounce buffers;
- copies only the requested subsection back into TileDB's buffer; and
- records intercepted opens, reads, requested bytes, aligned bytes, returned bytes, and failures.

It does not change writes, data preparation, GPU transfer, NCCL exchange, TileDB query planning, or
TileDB's own in-process caches. It is experimental benchmark tooling, not a production backend.
The benchmark enables it only after data preparation. Do not set
`WHOLEMEMORY_TILEDB_DIRECT_IO=1` yourself; `--direct-io` enables it at the correct point.

## Metrics and definitions

Every aggregate and synchronized raw sample now includes:

| Field | Meaning |
| --- | --- |
| `storage_read_ops` | Completed physical block-device reads during the sample |
| `storage_write_ops` | Completed physical block-device writes during the sample |
| `storage_total_io_ops` | Read plus write operations |
| `storage_read_iops` | Read operations divided by synchronized gather latency |
| `storage_write_iops` | Write operations divided by synchronized gather latency |
| `storage_iops` | Total operations divided by synchronized gather latency |
| `device_*` | Raw `/sys/class/block/DEVICE/stat` deltas and rates |
| `process_*` | Summed process syscall counts, retained as a fallback/diagnostic |
| `tiledb_vfs_read_ops` | TileDB logical VFS operations, when native stats are available |

`storage_*` uses the block-device counter when available. The counter observes the entire device,
so keep the node otherwise idle. A completed block-layer operation is not the same thing as a row,
TileDB range, TileDB tile, or necessarily one NVMe command; merging and splitting can change the
relationship.

Direct I/O results also retain `direct_io_read_ops`, `direct_io_requested_bytes`,
`direct_io_submitted_bytes`, and `direct_io_alignment_read_amplification`. These measure the shim
layer between TileDB's logical VFS requests and the physical device counters.

## 1. Update and rebuild

From the repository root, verify the branch and rebuild using the same environment and flags as the
preceding run:

```bash
git branch --show-current
git rev-parse HEAD
git status --short

cmake --build cpp/build -j"$(nproc)"
cmake --install cpp/build

export DIRECT_IO_PRELOAD="${CONDA_PREFIX}/lib/libwholememory_tiledb_direct_io_preload.so"
test -f "${DIRECT_IO_PRELOAD}"
ls -lh "${DIRECT_IO_PRELOAD}"
```

Substitute the actual build directory if it is not `cpp/build`. No root privileges are required.

## 2. Preflight the storage path

```bash
findmnt -T /raid -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
cat /sys/class/block/nvme1n1/queue/logical_block_size
cat /sys/class/block/nvme1n1/queue/physical_block_size
numactl --hardware
nvidia-smi topo -m
```

The data directory must resolve to the primary `/dev/nvme1n1` filesystem. Stop if it resolves to a
network filesystem or another device.

## 3. Run the focused matrix

Reuse the existing arrays, use a new result directory, and capture the complete log:

```bash
data_dir=/raid/abarghi/wholememory-tiledb-loading-20260827T194011Z/full
run_id=$(date -u +%Y%m%dT%H%M%SZ)
result_dir="/raid/abarghi/wholememory-tiledb-direct-io-iops-${run_id}"

DIRECT_IO_PRELOAD="${CONDA_PREFIX}/lib/libwholememory_tiledb_direct_io_preload.so" \
python/pylibwholegraph/benchmarks/run_tiledb_direct_io_iops_benchmark.sh \
  "${data_dir}" "${result_dir}" \
  2>&1 | tee "${result_dir}.log"
```

The default matrix contains:

| Result set | Purpose | Cases | Samples |
| --- | --- | ---: | ---: |
| `iops-overlap-clustered-width-2048` | Corrected 100k overlap curve | 21 | 105 |
| `iops-overlap-scattered-width-2048` | Random/scattered 25%-unique sentinel | 6 | 30 |
| `iops-overlap-multirun-width-2048` | Balanced 10/40/100-run headline matrix | 18 | 90 |
| `direct-io-multirun-width-2048` | Primary 40-run cross-rank Direct I/O case | 1 | 5 |

Expected validation:

```text
PASS: 46 aggregate configurations, 230 samples, IOPS complete, Direct I/O interception verified
```

The random sentinel remains because its IOPS profile may differ sharply from ICL-shaped access.
The 48k anchor and old 65,536-row continuity sentinel are omitted to keep this rerun focused.

## 4. Optional Direct I/O sensitivity

Run this only if the primary Direct I/O case succeeds and is useful. It adds the 10- and 100-run
cross-rank cases:

```bash
DIRECT_IO_SENSITIVITY=1 \
DIRECT_IO_PRELOAD="${CONDA_PREFIX}/lib/libwholememory_tiledb_direct_io_preload.so" \
python/pylibwholegraph/benchmarks/run_tiledb_direct_io_iops_benchmark.sh \
  "${data_dir}" "${result_dir}-sensitivity" \
  2>&1 | tee "${result_dir}-sensitivity.log"
```

That run should report 48 cases and 240 samples.

## 5. Verify the output

The validator runs automatically and can be repeated:

```bash
python3 python/pylibwholegraph/benchmarks/validate_tiledb_direct_io_iops_results.py \
  "${result_dir}"
```

Do not accept Direct I/O unless:

- the metadata says Direct I/O and the preload are enabled;
- every Direct I/O row uses `cache_mode=direct`;
- intercepted open and read counts are positive;
- open and read failure counts are zero;
- returned bytes equal requested bytes;
- aligned submitted bytes are at least requested bytes; and
- physical device read operations are positive.

The internal counters are the controlling proof. If diagnosis is needed, run one shortened
invocation under `strace -ff -e trace=openat,pread64` and confirm array opens contain `O_DIRECT`.

## 6. Interpret and preserve the result

Compare Direct I/O first with **buffered cold**, because both require storage reads. Buffered warm
remains the cached speed-of-light reference, not the expected Direct I/O latency.

Report end-to-end and query-submit latency, total operations and IOPS, read operations and IOPS,
physical and useful GiB/s, TileDB VFS operations when available, alignment amplification, CPU use,
RSS, and any change in host page-cache usage. Direct I/O may still be valuable if it preserves host
memory under TabICL model pressure even when it does not reduce latency.

Preserve JSON, aggregate/sample CSVs, logs, validator output, preflight output, and any trace. Do
not commit arrays or raw feature files.
