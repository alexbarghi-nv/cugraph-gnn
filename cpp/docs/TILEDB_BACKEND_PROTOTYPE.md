# Read-only TileDB backend prototype

This branch adds an optional TileDB storage location for read-only feature gathers. It deliberately
reuses WholeMemory's existing distributed partitioning and NCCL routing: each communicator rank
opens one local TileDB array, services the ids routed to its partition, and the global communicator
returns the gathered rows to requesting ranks.

## Build

Install the TileDB C/C++ library so that its `TileDBConfig.cmake` is on `CMAKE_PREFIX_PATH`, then
build with:

```bash
./build.sh libwholegraph pylibwholegraph cugraph-pyg --enable-tiledb
```

The default build remains unchanged and has no TileDB dependency. Calling the TileDB constructor in
a build without support returns `WHOLEMEMORY_NOT_SUPPORTED`.

## Rank-local array contract

Every rank's URI must identify a one-dimensional dense TileDB array with:

- an `INT64` dimension named `row`, with a zero-based domain large enough for the local partition;
- a fixed-sized `UINT8` attribute named `values`;
- `values.cell_val_num == row_stride * sizeof(dtype)`.

Rows in each local array are numbered from zero. WholeMemory continues to expose global row ids and
subtracts the owning rank's partition offset before querying TileDB. The URI passed by each rank may
be different. The Python convenience API replaces `{rank}` in a URI template with the global
communicator rank.

The build creates `wholememory_tiledb_ingest`, which converts one rank's contiguous row-major binary
file into this schema:

```bash
wholememory_tiledb_ingest \
  /mnt/nvme/features/rank_0.tdb rank_0.bin 10000000 512 4096 1048576 0
```

The arguments after the input file are row count, bytes per row, and optional TileDB tile extent in
rows. The next optional argument is the number of rows written per query. The final `0` or `1`
controls whether the resulting fragments are consolidated and vacuumed. Create one array for each
rank using the same partitioning passed to WholeMemory. Tile extent and consolidation are workload
parameters, not universal defaults: small extents reduce amplification for random gathers; larger
extents improve sequential bandwidth and metadata efficiency.

## Python usage

At the pylibwholegraph layer:

```python
features = pylibwholegraph.torch.create_wholememory_tensor_from_tiledb(
    comm,
    "/mnt/nvme/features/rank_{rank}.tdb",
    sizes=[total_rows, feature_width],
    dtype=torch.float32,
    tensor_entry_partition=rows_per_rank,
)
result = features.gather(cuda_indices)
```

At the cuGraph-PyG layer, a TileDB tensor can be installed into `FeatureStore` by reference:

```python
features = DistTensor(
    shape=(total_rows, feature_width),
    dtype=torch.float32,
    backend="tiledb",
    tiledb_uri="/mnt/nvme/features/rank_{rank}.tdb",
    partition_book=rows_per_rank,
)
feature_store.put_tensor(features, group_name="paper", attr_name="x")
```

## Buffer and execution path

The gather pipeline is:

1. bucket and exchange global ids on the existing CUDA stream;
2. copy the ids to a CUDA-pinned host buffer;
3. sort/deduplicate ids and coalesce adjacent ids into TileDB ranges;
4. ask TileDB to read full rows directly into a caller-owned pinned buffer;
5. restore duplicates and request order, including WholeMemory column slices, into a pinned buffer;
6. asynchronously stage the selected rows to a device send buffer;
7. use the existing NCCL all-to-all and CUDA reorder to produce the normal CUDA output tensor.

Pinned host allocation is therefore supported and used. This prototype does **not** pass host
pointers directly to NCCL: NCCL's documented user-buffer registration path describes CUDA/VMM or
`ncclMemAlloc` device allocations, not a portable pinned-host send contract. The explicit staging
copy is the compatibility baseline. It also means the first prototype does not reduce the peak
device send-buffer size versus the CPU backend; it removes resident feature storage from host/device
memory. A later optimization can chunk TileDB reads and communication to bound that staging buffer.

## Intentional prototype limits

- Read-only: scatter, optimizer updates, local tensor mapping, and file loading into an open handle
  are rejected.
- Distributed/NCCL handles only; hierarchy, VMM, NVSHMEM, and embedding-cache integration are out of
  scope.
- Gather output dtype must equal storage dtype. `force_dtype` conversion is not yet implemented.
- One array per global communicator rank. A node-shared array/local-communicator topology is a
  follow-up once the basic I/O path is measured.
- The TileDB query is synchronous with respect to the current CUDA stream. Overlap, prefetching,
  persistent pinned pools, and overlapped I/O/copies are follow-up performance work. Experimental
  bounded TileDB queries can be selected when creating a handle by setting
  `WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS`; zero or an unset variable preserves the unbounded default.
- Array schema is checked through query success and returned byte count. Rich up-front schema
  diagnostics can be added after the storage format stabilizes.

## GPU verification and benchmark checklist

Run the C++ TileDB storage tests first, then a one-rank gather, a multi-rank same-node gather, and a
multi-node gather. Cover duplicate/unsorted ids, empty gathers, rank boundaries, 1-D tensors,
subcolumn views, and invalid ids. Compare values against the CPU backend.

For performance, report random and locality-biased batches across several batch sizes and tile
extents. Capture application throughput, p50/p95 gather latency, NVMe bytes/read bandwidth, CPU
utilization, TileDB internal stats, H2D bandwidth, peak pinned memory, and peak device staging
memory. Compare with the pinned-CPU and internal NVMe implementations using identical partitions and
id traces.

`python/pylibwholegraph/benchmarks/tiledb_feature_fetch_benchmark.py` supports multiple local GPU
ranks, raw sample retention, block-device counters, TileDB statistics, staging phase timings,
recorded `.npy` ID traces, consolidated arrays, and query-chunk sweeps. Aggregate latency is the
slowest rank in each synchronized sample; aggregate throughput counts requested bytes from all
ranks. The phase metrics include `cpu_reorder_ms`, which isolates the final host-side scatter that
restores request order, expands duplicate IDs, and applies a WholeMemory column slice. It excludes
ID sorting/deduplication, TileDB range construction and query execution, and the H2D copy.
