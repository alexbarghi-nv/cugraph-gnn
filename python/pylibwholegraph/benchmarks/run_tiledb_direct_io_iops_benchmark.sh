#!/usr/bin/env bash
# Focused ICL-oriented IOPS rerun plus an experimental TileDB Direct I/O comparison.

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
direct_io_sensitivity=${DIRECT_IO_SENSITIVITY:-0}
direct_io_preload=${DIRECT_IO_PRELOAD:-${CONDA_PREFIX:-}/lib/libwholememory_tiledb_direct_io_preload.so}

if [[ ! -f "${direct_io_preload}" ]]; then
  echo "Direct I/O preload library not found: ${direct_io_preload}" >&2
  echo "Set DIRECT_IO_PRELOAD to libwholememory_tiledb_direct_io_preload.so" >&2
  exit 1
fi

mkdir -p "${data_dir}" "${output_dir}"

run_buffered() {
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

run_direct() {
  env \
    LD_PRELOAD="${direct_io_preload}" \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    WHOLEMEMORY_TILEDB_COMPUTE_CONCURRENCY="${compute_concurrency}" \
    WHOLEMEMORY_TILEDB_IO_CONCURRENCY="${io_concurrency}" \
    numactl --cpunodebind=1 --membind=1 \
    python3 "${benchmark_dir}/tiledb_feature_fetch_benchmark.py" \
      --data-dir "${data_dir}" \
      --world-size 4 \
      --rows "${rows}" \
      --width 2048 \
      --backends tiledb \
      --array-layouts node \
      --tile-extents 256 \
      --query-chunk-rows 0 \
      --cache-modes direct \
      --direct-io \
      --block-device /dev/nvme1n1 \
      "$@"
}

overlap_cases="independent,cross_rank_25,within_rank_25,combined_12_5,combined_6_25,combined_3_125,stress_1"
clustered_patterns="overlap_clustered_independent,overlap_clustered_cross_rank_25,overlap_clustered_within_rank_25,overlap_clustered_combined_12_5,overlap_clustered_combined_6_25,overlap_clustered_combined_3_125,overlap_clustered_stress_1"

direct_placements=clustered_runs_40
direct_patterns=overlap_clustered_runs_40_cross_rank_25
if [[ "${direct_io_sensitivity}" == "1" ]]; then
  direct_placements=clustered_runs_10,clustered_runs_40,clustered_runs_100
  direct_patterns=overlap_clustered_runs_10_cross_rank_25,overlap_clustered_runs_40_cross_rank_25,overlap_clustered_runs_100_cross_rank_25
fi

echo "Running the balanced ICL-shaped Direct I/O comparison first"
run_direct \
  --output "${output_dir}/direct-io-multirun-width-2048.json" \
  --overlap-cases cross_rank_25 \
  --overlap-placements "${direct_placements}" \
  --patterns "${direct_patterns}" \
  --batch-sizes 100000 \
  --warmup 2 \
  --repetitions 5

echo "Rerunning the 100k clustered overlap curve with IOPS counters"
run_buffered \
  --output "${output_dir}/iops-overlap-clustered-width-2048.json" \
  --overlap-cases "${overlap_cases}" \
  --overlap-placements clustered \
  --patterns "${clustered_patterns}" \
  --batch-sizes 100000 \
  --warmup 2 \
  --repetitions 5 \
  --storage-baseline

echo "Rerunning the scattered 25%-unique sentinel with IOPS counters"
run_buffered \
  --output "${output_dir}/iops-overlap-scattered-width-2048.json" \
  --overlap-cases cross_rank_25,within_rank_25 \
  --overlap-placements scattered \
  --patterns overlap_scattered_cross_rank_25,overlap_scattered_within_rank_25 \
  --batch-sizes 100000 \
  --warmup 2 \
  --repetitions 5

echo "Rerunning the balanced ICL-shaped multi-run matrix with IOPS counters"
run_buffered \
  --output "${output_dir}/iops-overlap-multirun-width-2048.json" \
  --overlap-cases cross_rank_25,within_rank_25 \
  --overlap-placements clustered_runs_10,clustered_runs_40,clustered_runs_100 \
  --patterns overlap_clustered_runs_10_cross_rank_25,overlap_clustered_runs_10_within_rank_25,overlap_clustered_runs_40_cross_rank_25,overlap_clustered_runs_40_within_rank_25,overlap_clustered_runs_100_cross_rank_25,overlap_clustered_runs_100_within_rank_25 \
  --batch-sizes 100000 \
  --warmup 2 \
  --repetitions 5

python3 "${benchmark_dir}/validate_tiledb_direct_io_iops_results.py" \
  "${output_dir}"

echo "Direct I/O and IOPS benchmark complete: ${output_dir}"
