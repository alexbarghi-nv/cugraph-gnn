#!/usr/bin/env bash
# Focused single-node TabICLv2-oriented overlap benchmark for TileDB and pinned CPU.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

data_dir=$1
output_dir=$2
benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

rows=${ROWS:-16777216}
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
      --width 2048 \
      --backends cpu,tiledb \
      --array-layouts node \
      --tile-extents 256 \
      --query-chunk-rows 0 \
      --cache-modes cold,warm \
      --block-device /dev/nvme1n1 \
      "$@"
}

overlap_cases="independent,cross_rank_25,within_rank_25,combined_12_5,combined_6_25,combined_3_125,stress_1"
clustered_patterns="overlap_clustered_independent,overlap_clustered_cross_rank_25,overlap_clustered_within_rank_25,overlap_clustered_combined_12_5,overlap_clustered_combined_6_25,overlap_clustered_combined_3_125,overlap_clustered_stress_1"

# Decision matrix: seven exact overlap topologies at the native and large TabICLv2 anchors.
echo "Running clustered overlap matrix"
run_benchmark \
  --output "${output_dir}/tabicl-overlap-clustered-width-2048.json" \
  --overlap-cases "${overlap_cases}" \
  --overlap-placements clustered \
  --patterns "${clustered_patterns}" \
  --batch-sizes 48000,100000 \
  --warmup 2 \
  --repetitions 5 \
  --storage-baseline

# Scattered sentinel: hold node-wide uniqueness at 25% while changing its source.
echo "Running scattered 25%-unique topology pair"
run_benchmark \
  --output "${output_dir}/tabicl-overlap-scattered-width-2048.json" \
  --overlap-cases cross_rank_25,within_rank_25 \
  --overlap-placements scattered \
  --patterns overlap_scattered_cross_rank_25,overlap_scattered_within_rank_25 \
  --batch-sizes 100000 \
  --warmup 2 \
  --repetitions 5

# Preserve a direct comparison with the preceding 65,536-row locality measurements.
echo "Running 65,536-row continuity sentinel"
run_benchmark \
  --output "${output_dir}/tabicl-continuity-width-2048.json" \
  --locality-window-rows 256,4096 \
  --patterns window_256,window_4096 \
  --batch-sizes 65536 \
  --warmup 2 \
  --repetitions 5

echo "TabICLv2-oriented overlap benchmark complete: ${output_dir}"
