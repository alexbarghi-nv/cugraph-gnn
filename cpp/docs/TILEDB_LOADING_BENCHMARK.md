# Single-node TileDB loading benchmark

This benchmark measures read-only TileDB-backed WholeMemory against distributed host-pinned
(`cpu`) WholeMemory as the in-memory speed-of-light reference. The goal is to minimize the
out-of-core latency premium while enabling feature tables that cannot coexist in memory with large
models, not to require TileDB to outperform an all-resident table. It deliberately contains no
training, sampling, or second communicator. Four
processes use one NCCL/WholeMemory communicator and visible GPUs 4-7. The launch script binds the
processes and their memory allocations to CPU socket/NUMA node 1, which is local to the GPUs and
`/dev/nvme1n1` on the RTX PRO 6000 test node.

For machine setup, preflight gates, smoke testing, restart behavior, and result handoff, follow
[TILEDB_LOADING_BENCHMARK_RUNBOOK.md](TILEDB_LOADING_BENCHMARK_RUNBOOK.md).
For the focused physical-IOPS rerun and experimental Direct I/O comparison, follow
[TILEDB_DIRECT_IO_IOPS_RUNBOOK.md](TILEDB_DIRECT_IO_IOPS_RUNBOOK.md).

Run the complete width matrix with:

```bash
python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh \
  /raid/abarghi/wholememory-tiledb-loading \
  /raid/abarghi/wholememory-tiledb-loading-results
```

For the next TabICLv2-oriented overlap study, use the focused overlap runner:

```bash
python/pylibwholegraph/benchmarks/run_tiledb_tabicl_overlap_benchmark.sh \
  /raid/abarghi/wholememory-tiledb-loading \
  /raid/abarghi/wholememory-tiledb-tabicl-overlap-results
```

It fixes width 2,048, a 256-row tile extent, one node-shared array, request sizes 48,000 and
100,000 per rank, and the pinned-CPU reference. Seven exact clustered overlap cases separate
within-rank repetition from cross-rank sharing. A corrected scattered 100,000-row sentinel compares
the two ways to obtain 25% node-wide uniqueness using the same physical unique-ID set. Additional
10-, 40-, and 100-run clustered pairs distribute those same 25%-unique contexts across owner
partitions. The preceding 65,536-row window cases are retained only as a continuity check. The run
contains 72 configurations, 360 measured samples, and 504 synchronized rounds including warmups.

The overlap cases are generated from distinct unique IDs before deliberate repetition is added:

| Case | Within-rank repetition | Cross-rank sharing | Node-wide unique fraction |
| --- | ---: | --- | ---: |
| `independent` | 1x | none | 100% |
| `cross_rank_25` | 1x | all four ranks | 25% |
| `within_rank_25` | 4x | none | 25% |
| `combined_12_5` | 2x | all four ranks | 12.5% |
| `combined_6_25` | 4x | all four ranks | 6.25% |
| `combined_3_125` | 8x | all four ranks | 3.125% |
| `stress_1` | 25x | all four ranks | 1% |

`clustered` places the node-wide unique set in one contiguous span; `scattered` applies a
deterministic bijection across the global row space. `clustered_runs_10`, `clustered_runs_40`, and
`clustered_runs_100` create that number of non-overlapping contiguous runs, distribute them across
WholeMemory owner partitions, and balance unique-row ownership. For each placement and sample,
`cross_rank_25` and `within_rank_25` derive from the identical node-wide unique IDs; only assignment
and repetition across ranks changes. These are controlled storage diagnostics, not claims about
the final TabICLv2 distribution. Time-series ordering and application-derived grouped contexts are
still deferred until an end-to-end trace is available.

The aggregate and sample outputs add within-rank and node-wide unique fractions, mean repetition,
requesting ranks per unique row, owner-rank max/mean imbalance, the exact node-wide unique-ID SHA-256
digest, node-wide range count, and node-wide tile count. The digest and physical counts must match
between paired topologies. Existing range, tile, planning, read, H2D, GPU expansion, NCCL, and
output-reorder measurements remain unchanged.

The unqualified phase columns remain the slowest end-to-end rank for backward compatibility. Raw
samples additionally record the rank maximum, rank mean, max-rank identity, storage-owner maximum,
and storage-owner mean for every phase timing and count. Aggregate rows report the mean of those
rank-aware sample values. Use the rank-aware fields for diagnosis because the slowest rank may be a
waiting non-owner with no storage work.

The preceding GPU-compaction validation can still be reproduced with:

```bash
python/pylibwholegraph/benchmarks/run_tiledb_gpu_compaction_benchmark.sh \
  /raid/abarghi/wholememory-tiledb-loading \
  /raid/abarghi/wholememory-tiledb-gpu-compaction-results
```

That matrix keeps all three widths but uses one node-shared array, 256- and 4,096-row tiles,
8,192- and 65,536-row batches, 256- and 4,096-row locality windows, both cache modes, two warmups,
and five measurements. It adds one width-2,048 random sentinel and one width-2,048 node/rank spot
check. At the defaults this is 71 configurations, 349 measured samples, and 488 synchronized gather
rounds, compared with 468 configurations, 4,680 samples, and 6,084 rounds in the complete matrix.
Existing arrays are reused when their benchmark marker matches, so point the launcher at the prior
full-run data directory and use a new output directory.

The default matrix covers:

- vector widths 128, 512, and 2,048 float32 values;
- random IDs and IDs drawn from one 256-, 4,096-, or 65,536-row window;
- cold and page-cache-warm TileDB reads;
- one rank-local array per WholeMemory rank and one node-shared array using global row coordinates;
- tile extents 256, 4,096, and 65,536 rows;
- batch sizes 1,024, 8,192, and 65,536 rows per rank;
- the resident distributed host-pinned WholeMemory baseline for every trace and batch size.

At the default row count, the three raw width datasets total 168 GiB. Keeping rank-local and
node-shared arrays for all three tile extents requires six additional copies, so budget roughly
1,176 GiB plus TileDB metadata and any temporary consolidation space.

Environment variables can reduce the run without changing the script:

```bash
ROWS=1048576 WIDTHS="128" REPETITIONS=3 WARMUP=1 \
  python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh DATA_DIR OUTPUT_DIR
```

The underlying benchmark also accepts `--patterns` and `--cache-modes`, allowing a caller to avoid
the default random/locality and cold/warm Cartesian products.

`TILEDB_COMPUTE_CONCURRENCY` and `TILEDB_IO_CONCURRENCY` default to 8 per rank. Four ranks therefore
avoid creating four full-machine TileDB worker pools. Both values and the inherited CPU affinity
are recorded in the result metadata.

## Measurements

Every synchronized sample retains end-to-end latency, useful throughput, physical block-device
reads, read amplification, CPU time, RSS, and peak CUDA temporary allocation. TileDB gathers also
record these non-overlapping implementation boundaries where possible:

1. WholeMemory ID routing and exchange;
2. GPU ID sort;
3. GPU deduplication and inverse-map construction;
4. compact pinned staging allocation;
5. unique-ID D2H;
6. ID decode/global-to-array-coordinate conversion;
7. TileDB query allocation/setup;
8. adjacent-ID range construction and subarray insertion;
9. `tiledb_query_submit`, which encloses TileDB planning, tile I/O, unfiltering, and result copying;
10. optional CPU copy for WholeMemory column slices (zero for a full-row gather);
11. unique-row H2D;
12. GPU duplicate expansion into owner send-buffer order;
13. WholeMemory embedding exchange;
14. final GPU output reorder.

With `--tiledb-stats`, each sample additionally records a compact selection of TileDB's nested
internal timers: tile-overlap planning, relevant-tile overlap, subarray partitioning, tile reads,
unfiltering, fixed-cell copying, and total reader work. These internal timers overlap and must not be
summed. The result also retains TileDB range, tile, VFS-operation, and VFS-byte counters.

Cold-cache control uses `POSIX_FADV_DONTNEED`, which does not require root but is an eviction hint.
The measured `/dev/nvme1n1` sector deltas are therefore the authoritative check that a cold sample
actually reached storage. Warm cases read the complete array into the page cache before timing.

The benchmark also records completed block-device read and write operations. Aggregate and raw
sample outputs include total I/O operations and read, write, and total IOPS. These are physical
block-layer measurements from `/sys/class/block/DEVICE/stat` when available; summed process syscall
counts are retained as a fallback. They are distinct from TileDB logical VFS operations, ranges,
and tiles.

An experimental `--direct-io` mode is available through
`libwholememory_tiledb_direct_io_preload.so`. It applies `O_DIRECT` only after data preparation,
repairs unaligned TileDB reads with aligned bounce buffers, and reports operation and byte counters.
See `TILEDB_DIRECT_IO_IOPS_RUNBOOK.md` for the focused comparison and acceptance checks.

The script produces one JSON, aggregate CSV, raw-sample CSV, and per-rank checkpoint per vector
width. The benchmark intentionally does not generate a report artifact; the CSVs are the primary
comparison surface.
