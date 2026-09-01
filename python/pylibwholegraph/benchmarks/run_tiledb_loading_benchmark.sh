#!/usr/bin/env bash
# Run the single-node loading matrix on RTX GPUs 4-7 and NUMA node 1.

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
repetitions=${REPETITIONS:-10}
warmup=${WARMUP:-3}
gpu_ids=${GPU_IDS:-"4,5,6,7"}
compute_concurrency=${TILEDB_COMPUTE_CONCURRENCY:-8}
io_concurrency=${TILEDB_IO_CONCURRENCY:-8}

mkdir -p "${data_dir}" "${output_dir}"

for width in ${widths}; do
  echo "Running width=${width} on GPUs ${gpu_ids}, NUMA node 1"
  env \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    WHOLEMEMORY_TILEDB_COMPUTE_CONCURRENCY="${compute_concurrency}" \
    WHOLEMEMORY_TILEDB_IO_CONCURRENCY="${io_concurrency}" \
    numactl --cpunodebind=1 --membind=1 \
    python3 "${benchmark_dir}/tiledb_feature_fetch_benchmark.py" \
      --data-dir "${data_dir}" \
      --output "${output_dir}/loading-width-${width}.json" \
      --world-size 4 \
      --rows "${rows}" \
      --width "${width}" \
      --backends cpu,tiledb \
      --array-layouts rank,node \
      --batch-sizes 1024,8192,65536 \
      --locality-window-rows 256,4096,65536 \
      --tile-extents 256,4096,65536 \
      --query-chunk-rows 0 \
      --warmup "${warmup}" \
      --repetitions "${repetitions}" \
      --block-device /dev/nvme1n1 \
      --tiledb-stats \
      --storage-baseline
done
