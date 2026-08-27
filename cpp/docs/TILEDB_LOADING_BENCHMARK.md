# Single-node TileDB loading benchmark

This benchmark compares read-only TileDB-backed WholeMemory with distributed host-pinned
(`cpu`) WholeMemory. It deliberately contains no training, sampling, or second communicator. Four
processes use one NCCL/WholeMemory communicator and visible GPUs 4-7. The launch script binds the
processes and their memory allocations to CPU socket/NUMA node 1, which is local to the GPUs and
`/dev/nvme1n1` on the RTX PRO 6000 test node.

For machine setup, preflight gates, smoke testing, restart behavior, and result handoff, follow
[TILEDB_LOADING_BENCHMARK_RUNBOOK.md](TILEDB_LOADING_BENCHMARK_RUNBOOK.md).

Run the complete width matrix with:

```bash
python/pylibwholegraph/benchmarks/run_tiledb_loading_benchmark.sh \
  /raid/abarghi/wholememory-tiledb-loading \
  /raid/abarghi/wholememory-tiledb-loading-results
```

For GPU-compaction validation and TabICLv2-like locality, prefer the focused runner:

```bash
python/pylibwholegraph/benchmarks/run_tiledb_gpu_compaction_benchmark.sh \
  /raid/abarghi/wholememory-tiledb-loading \
  /raid/abarghi/wholememory-tiledb-gpu-compaction-results
```

Its primary matrix keeps all three widths but uses one node-shared array, 256- and 4,096-row tiles,
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

The script produces one JSON, aggregate CSV, raw-sample CSV, and per-rank checkpoint per vector
width. The benchmark intentionally does not generate a report artifact; the CSVs are the primary
comparison surface.
