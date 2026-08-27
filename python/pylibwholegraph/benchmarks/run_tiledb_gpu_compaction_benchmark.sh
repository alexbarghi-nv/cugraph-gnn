#!/usr/bin/env bash
# Focused single-node validation of TileDB GPU ID compaction and duplicate expansion.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

data_dir=$1
output_dir=$2
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

rows=${ROWS:-16777216}
widths=${WIDTHS:-"128 512 2048"}
gpu_ids=${GPU_IDS:-"4,5,6,7"}
compute_concurrency=${TILEDB_COMPUTE_CONCURRENCY:-8}
io_concurrency=${TILEDB_IO_CONCURRENCY:-8}

mkdir -p "${data_dir}" "${output_dir}"

run_benchmark() {
  env \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    WHOLEMEMORY_TILEDB_COMPUTE_CONCURRENCY="${compute_concurrency}" \
    WHOLEMEMORY_TILEDB_IO_CONCURRENCY="${io_concurrency}" \
    numactl --cpunodebind=1 --membind=1 \
    python3 "${benchmark_dir}/tiledb_feature_fetch_benchmark.py" \
      --data-dir "${data_dir}" \
      --world-size 4 \
      --rows "${rows}" \
      --query-chunk-rows 0 \
      --block-device /dev/nvme1n1 \
      "$@"
}

# Primary locality matrix: 60 configurations, 300 measured samples, and 420 rounds including
# warmups across the default three widths.
for width in ${widths}; do
  echo "Running focused locality matrix for width=${width}"
  run_benchmark \
    --output "${output_dir}/gpu-compaction-locality-width-${width}.json" \
    --width "${width}" \
    --backends cpu,tiledb \
    --array-layouts node \
    --tile-extents 256,4096 \
    --locality-window-rows 256,4096 \
    --patterns window_256,window_4096 \
    --cache-modes cold,warm \
    --batch-sizes 8192,65536 \
    --warmup 2 \
    --repetitions 5 \
    --storage-baseline
done

# One deliberately difficult random sentinel rather than a random Cartesian product.
echo "Running width=2048 random sentinel"
run_benchmark \
  --output "${output_dir}/gpu-compaction-random-width-2048.json" \
  --width 2048 \
  --backends cpu,tiledb \
  --array-layouts node \
  --tile-extents 256 \
  --locality-window-rows 256 \
  --patterns random \
  --cache-modes cold,warm \
  --batch-sizes 65536 \
  --warmup 1 \
  --repetitions 3

# Direct layout comparison at the widest vectors and largest batch.
echo "Running width=2048 node/rank layout spot check"
run_benchmark \
  --output "${output_dir}/gpu-compaction-layout-width-2048.json" \
  --width 2048 \
  --backends tiledb \
  --array-layouts rank,node \
  --tile-extents 256 \
  --locality-window-rows 256,4096 \
  --patterns window_256,window_4096 \
  --cache-modes cold,warm \
  --batch-sizes 65536 \
  --warmup 2 \
  --repetitions 5

echo "Focused GPU-compaction benchmark complete: ${output_dir}"
