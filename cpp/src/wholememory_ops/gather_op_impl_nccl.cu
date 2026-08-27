/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <chrono>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/functions/sort_indices_func.h"
#include "wholememory_ops/gather_op_impl.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace {

thread_local wholememory_tiledb_gather_metrics_t last_tiledb_gather_metrics{};

double elapsed_ms(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
    .count();
}

}  // namespace

extern "C" wholememory_error_code_t wholememory_get_last_tiledb_gather_metrics(
  wholememory_tiledb_gather_metrics_t* metrics)
{
  if (metrics == nullptr) { return WHOLEMEMORY_INVALID_INPUT; }
  *metrics = last_tiledb_gather_metrics;
  return WHOLEMEMORY_SUCCESS;
}

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_nccl(wholememory_handle_t wholememory_handle,
                                                 wholememory_matrix_description_t wholememory_desc,
                                                 void* indices,
                                                 wholememory_array_description_t indice_desc,
                                                 void* output,
                                                 wholememory_matrix_description_t output_desc,
                                                 wholememory_env_func_t* p_env_fns,
                                                 cudaStream_t stream,
                                                 int gather_sms)
{
  try {
    last_tiledb_gather_metrics = {};
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[1] > wholememory_desc.stride) {
      return WHOLEMEMORY_INVALID_INPUT;
    }

    wm_thrust_allocator thrust_allocator(p_env_fns);

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;

    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));

    temp_memory_handle host_rank_id_count(p_env_fns), host_recv_rank_id_count(p_env_fns);
    int64_t* host_rank_id_count_ptr =
      static_cast<int64_t*>(host_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));
    int64_t* host_recv_rank_id_count_ptr =
      static_cast<int64_t*>(host_recv_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));

    temp_memory_handle dev_recv_indice_buffer(p_env_fns);
    temp_memory_handle dev_raw_indice(p_env_fns);
    int64_t* dev_raw_indice_ptr =
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT64));

    int64_t total_recv_count = 0;

    temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
    temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));

    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_rank_partition_offsets(host_embedding_entry_offsets_ptr, wholememory_handle));
    auto const memory_location = wholememory_get_memory_location(wholememory_handle);
    for (int i = 0; i < world_size + 1; i++) {
      size_t offset = host_embedding_entry_offsets_ptr[i];
      WHOLEMEMORY_EXPECTS_NOTHROW(
        offset % embedding_entry_size == 0,
        "embedding memory offset of rank%d=%ld is not multiple of embedding_entry_size=%ldx%ld",
        i,
        offset,
        element_size,
        wholememory_desc.stride);
      host_embedding_entry_offsets_ptr[i] /= embedding_entry_size;
    }

    WM_CUDA_CHECK(cudaMemcpyAsync(dev_embedding_entry_offsets_ptr,
                                  host_embedding_entry_offsets_ptr,
                                  (world_size + 1) * sizeof(size_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    auto const routing_start =
      memory_location == WHOLEMEMORY_ML_TILEDB ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    WHOLEMEMORY_RETURN_ON_FAIL(bucket_and_exchange_ids_func(indices,
                                                            indice_desc,
                                                            host_recv_rank_id_count_ptr,
                                                            host_rank_id_count_ptr,
                                                            &dev_recv_indice_buffer,
                                                            dev_raw_indice_ptr,
                                                            dev_embedding_entry_offsets_ptr,
                                                            wm_comm,
                                                            &thrust_allocator,
                                                            p_env_fns,
                                                            stream));
    double id_routing_ms = 0.0;
    if (memory_location == WHOLEMEMORY_ML_TILEDB) {
      // The TileDB path is already synchronous before its CPU read. Synchronizing here exposes a
      // real routing phase without removing overlap that currently exists in this path.
      WM_CUDA_CHECK(cudaStreamSynchronize(stream));
      id_routing_ms = elapsed_ms(routing_start);
    }
    // Local Gather
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    size_t local_mem_offset, local_mem_size;
    temp_memory_handle dev_local_gather_buffer(p_env_fns);
    temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      wholememory_desc.sizes[1] * total_recv_count, output_desc.dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      wholememory_desc.sizes[1] * indice_desc.size, output_desc.dtype);
    int64_t local_buffer_size[2] = {total_recv_count, wholememory_desc.sizes[1]};
    wholememory_matrix_description_t local_gather_buffer_desc = wholememory_create_matrix_desc(
      local_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
    auto dev_recv_indice_desc =
      wholememory_create_array_desc(total_recv_count, 0, indice_desc.dtype);
    if (memory_location == WHOLEMEMORY_ML_TILEDB) {
      // TileDB reads caller-owned buffers. Use page-locked buffers so the only required device
      // staging copy is asynchronous and explicit; the public gather result remains a CUDA tensor.
      if (wholememory_desc.dtype != output_desc.dtype) {
        WHOLEMEMORY_ERROR("TileDB gather does not yet support output dtype conversion");
        return WHOLEMEMORY_NOT_SUPPORTED;
      }
      last_tiledb_gather_metrics.valid         = 1;
      last_tiledb_gather_metrics.id_routing_ms = id_routing_ms;
      auto phase_start                         = std::chrono::steady_clock::now();
      temp_memory_handle dev_tiledb_sorted_indices(p_env_fns);
      temp_memory_handle dev_tiledb_sorted_positions(p_env_fns);
      temp_memory_handle dev_tiledb_unique_indices(p_env_fns);
      temp_memory_handle dev_tiledb_inverse_indices(p_env_fns);
      void* dev_tiledb_sorted_indices_ptr =
        dev_tiledb_sorted_indices.device_malloc(total_recv_count, indice_desc.dtype);
      void* dev_tiledb_sorted_positions_ptr =
        dev_tiledb_sorted_positions.device_malloc(total_recv_count, indice_desc.dtype);
      void* dev_tiledb_unique_indices_ptr =
        dev_tiledb_unique_indices.device_malloc(total_recv_count, indice_desc.dtype);
      auto* dev_tiledb_inverse_indices_ptr = static_cast<int64_t*>(
        dev_tiledb_inverse_indices.device_malloc(total_recv_count, WHOLEMEMORY_DT_INT64));

      int64_t unique_recv_count = 0;
      if (total_recv_count > 0) {
        // TileDB has a high per-row storage cost compared with resident memory. Compact on the
        // owner GPU so the CPU query and both host transfers contain each requested row once.
        phase_start = std::chrono::steady_clock::now();
        WHOLEMEMORY_RETURN_ON_FAIL(sort_indices_func(dev_recv_indice_buffer.pointer(),
                                                     dev_recv_indice_desc,
                                                     dev_tiledb_sorted_indices_ptr,
                                                     dev_tiledb_sorted_positions_ptr,
                                                     &thrust_allocator,
                                                     p_env_fns,
                                                     stream));
        WM_CUDA_CHECK(cudaStreamSynchronize(stream));
        last_tiledb_gather_metrics.gpu_sort_ms = elapsed_ms(phase_start);

        phase_start = std::chrono::steady_clock::now();
        WHOLEMEMORY_RETURN_ON_FAIL(compact_sorted_unique_indices_func(
          dev_tiledb_sorted_indices_ptr,
          dev_tiledb_sorted_positions_ptr,
          dev_recv_indice_desc,
          dev_tiledb_unique_indices_ptr,
          dev_tiledb_inverse_indices_ptr,
          &unique_recv_count,
          &thrust_allocator,
          stream));
        last_tiledb_gather_metrics.gpu_deduplicate_ms = elapsed_ms(phase_start);
      }

      phase_start = std::chrono::steady_clock::now();
      temp_memory_handle host_tiledb_indices(p_env_fns);
      temp_memory_handle host_tiledb_raw_rows(p_env_fns);
      temp_memory_handle host_tiledb_gather_rows(p_env_fns);
      temp_memory_handle dev_tiledb_unique_rows(p_env_fns);
      auto const output_row_bytes =
        wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(output_desc.dtype);
      auto const gather_bytes = unique_recv_count * output_row_bytes;
      auto const direct_full_row_read = wholememory_desc.storage_offset == 0 &&
                                        output_row_bytes == embedding_entry_size;
      void* host_tiledb_indices_ptr =
        host_tiledb_indices.pinned_malloc(unique_recv_count, indice_desc.dtype);
      void* host_tiledb_gather_rows_ptr = host_tiledb_gather_rows.pinned_malloc(
        unique_recv_count * wholememory_desc.sizes[1], output_desc.dtype);
      void* dev_tiledb_unique_rows_ptr =
        dev_tiledb_unique_rows.device_malloc(unique_recv_count * wholememory_desc.sizes[1],
                                             output_desc.dtype);
      auto const raw_staging_bytes = direct_full_row_read
                                       ? gather_bytes
                                       : unique_recv_count * embedding_entry_size;
      void* host_tiledb_raw_rows_ptr =
        direct_full_row_read
          ? host_tiledb_gather_rows_ptr
          : host_tiledb_raw_rows.pinned_malloc(raw_staging_bytes, WHOLEMEMORY_DT_INT8);
      last_tiledb_gather_metrics.staging_allocation_ms = elapsed_ms(phase_start);

      auto const index_bytes =
        unique_recv_count * wholememory_dtype_get_element_size(indice_desc.dtype);
      last_tiledb_gather_metrics.index_bytes = index_bytes;
      if (index_bytes > 0) {
        phase_start = std::chrono::steady_clock::now();
        WM_CUDA_CHECK(cudaMemcpyAsync(host_tiledb_indices_ptr,
                                      dev_tiledb_unique_indices_ptr,
                                      index_bytes,
                                      cudaMemcpyDeviceToHost,
                                      stream));
        // TileDB is a CPU API and must not observe the ids until bucket/exchange and D2H finish.
        WM_CUDA_CHECK(cudaStreamSynchronize(stream));
        last_tiledb_gather_metrics.indices_d2h_ms = elapsed_ms(phase_start);
      }

      last_tiledb_gather_metrics.raw_staging_bytes = raw_staging_bytes;
      phase_start                                  = std::chrono::steady_clock::now();
      WHOLEMEMORY_RETURN_ON_FAIL(
        wholememory::tiledb_read_rows_from_handle(wholememory_handle,
                                                  host_tiledb_indices_ptr,
                                                  indice_desc.dtype,
                                                  unique_recv_count,
                                                  wholememory_desc.storage_offset * element_size,
                                                  output_row_bytes,
                                                  host_tiledb_raw_rows_ptr,
                                                  raw_staging_bytes,
                                                  host_tiledb_gather_rows_ptr,
                                                  true,
                                                  &last_tiledb_gather_metrics));
      last_tiledb_gather_metrics.tiledb_read_ms = elapsed_ms(phase_start);
      last_tiledb_gather_metrics.storage_requested_rows = total_recv_count;

      last_tiledb_gather_metrics.output_bytes = gather_bytes;
      if (gather_bytes > 0) {
        phase_start = std::chrono::steady_clock::now();
        WM_CUDA_CHECK(cudaMemcpyAsync(dev_tiledb_unique_rows_ptr,
                                      host_tiledb_gather_rows_ptr,
                                      gather_bytes,
                                      cudaMemcpyHostToDevice,
                                      stream));
        // Keep the pinned allocation alive through the copy. NCCL remains ordered after it on the
        // same stream, and receives into the existing device buffer.
        WM_CUDA_CHECK(cudaStreamSynchronize(stream));
        last_tiledb_gather_metrics.rows_h2d_ms = elapsed_ms(phase_start);

        phase_start = std::chrono::steady_clock::now();
        auto unique_rows_desc = local_gather_buffer_desc;
        unique_rows_desc.sizes[0] = unique_recv_count;
        auto inverse_desc =
          wholememory_create_array_desc(total_recv_count, 0, WHOLEMEMORY_DT_INT64);
        auto unique_rows_gref =
          wholememory_create_continuous_global_reference(dev_tiledb_unique_rows_ptr);
        // Restore the exact post-routing order expected by the unchanged NCCL exchange.
        WHOLEMEMORY_RETURN_ON_FAIL(gather_func(unique_rows_gref,
                                               unique_rows_desc,
                                               dev_tiledb_inverse_indices_ptr,
                                               inverse_desc,
                                               dev_local_gather_buffer_ptr,
                                               local_gather_buffer_desc,
                                               stream,
                                               gather_sms));
        WM_CUDA_CHECK(cudaStreamSynchronize(stream));
        last_tiledb_gather_metrics.gpu_expand_ms = elapsed_ms(phase_start);
      }
    } else {
      void* local_fake_ptr = nullptr;
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
        &local_fake_ptr, &local_mem_size, &local_mem_offset, wholememory_handle));
      local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
      wholememory_gref_t local_fake_gref =
        wholememory_create_continuous_global_reference(local_fake_ptr);
      WHOLEMEMORY_RETURN_ON_FAIL(gather_func(local_fake_gref,
                                             wholememory_desc,
                                             dev_recv_indice_buffer.pointer(),
                                             dev_recv_indice_desc,
                                             dev_local_gather_buffer_ptr,
                                             local_gather_buffer_desc,
                                             stream,
                                             gather_sms));
    }
    // AllToAllV for embeddings
    size_t embedding_size =
      wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(output_desc.dtype);
    auto const embedding_exchange_start =
      memory_location == WHOLEMEMORY_ML_TILEDB ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             host_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wm_comm,
                                                             stream));
    if (memory_location == WHOLEMEMORY_ML_TILEDB) {
      WM_CUDA_CHECK(cudaStreamSynchronize(stream));
      last_tiledb_gather_metrics.embedding_exchange_ms = elapsed_ms(embedding_exchange_start);
    }
    // Local reorder
    int64_t total_need_indice_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_need_indice_count += host_rank_id_count_ptr[i];
    }
    wholememory_gref_t output_gref = wholememory_create_continuous_global_reference(output);
    wholememory_matrix_description_t local_recv_buffer_desc =
      wholememory_create_matrix_desc(output_desc.sizes, output_desc.sizes[1], 0, output_desc.dtype);
    local_recv_buffer_desc.sizes[0] = total_need_indice_count;
    auto raw_indice_desc =
      wholememory_create_array_desc(total_need_indice_count, 0, WHOLEMEMORY_DT_INT64);
    auto const output_reorder_start =
      memory_location == WHOLEMEMORY_ML_TILEDB ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    WHOLEMEMORY_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            local_recv_buffer_desc,
                                            dev_raw_indice_ptr,
                                            raw_indice_desc,
                                            output_gref,
                                            output_desc,
                                            stream));
    WM_CUDA_CHECK(cudaGetLastError());
    if (memory_location == WHOLEMEMORY_ML_TILEDB) {
      WM_CUDA_CHECK(cudaStreamSynchronize(stream));
      last_tiledb_gather_metrics.output_reorder_ms = elapsed_ms(output_reorder_start);
    }
    // WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_gather_distributed(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
#ifdef WITH_NVSHMEM_SUPPORT

  if (wholememory_get_distributed_backend(wholememory_handle) == WHOLEMEMORY_DB_NVSHMEM) {
    return wholememory_gather_nvshmem(wholememory_handle,
                                      wholememory_desc,
                                      indices,
                                      indice_desc,
                                      output,
                                      output_desc,
                                      p_env_fns,
                                      stream,
                                      gather_sms);
  }
#endif
  return wholememory_gather_nccl(wholememory_handle,
                                 wholememory_desc,
                                 indices,
                                 indice_desc,
                                 output,
                                 output_desc,
                                 p_env_fns,
                                 stream,
                                 gather_sms);
}
}  // namespace wholememory_ops
