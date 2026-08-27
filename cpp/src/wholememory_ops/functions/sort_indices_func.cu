/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sort_indices_func.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/sequence.h>

#include <algorithm>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename IndexT>
struct UnsignedType {};

template <>
struct UnsignedType<int> {
  using UType = unsigned int;
};

template <>
struct UnsignedType<int64_t> {
  using UType = uint64_t;
};

template <typename IndexT>
void sort_indices_temp_func(const void* indices_before_sort,
                            wholememory_array_description_t indices_desc,
                            void* indices_after_sort,
                            void* raw_indices,
                            wm_thrust_allocator* p_thrust_allocator,
                            wholememory_env_func_t* p_env_fns,
                            cudaStream_t stream)
{
  auto index_type = indices_desc.dtype;
  WHOLEMEMORY_CHECK(indices_desc.storage_offset == 0);
  WHOLEMEMORY_CHECK(index_type == WHOLEMEMORY_DT_INT || index_type == WHOLEMEMORY_DT_INT64);
  wm_thrust_allocator& allocator = *p_thrust_allocator;

  IndexT* seq_indices = reinterpret_cast<IndexT*>(allocator.allocate(
    wholememory_get_memory_element_count_from_array(&indices_desc) * sizeof(IndexT)));
  thrust::sequence(thrust::cuda::par_nosync(allocator).on(stream),
                   seq_indices,
                   seq_indices + indices_desc.size,
                   0);
  // use UTypeT to put minus indices at last.
  using UTypeT                  = typename UnsignedType<IndexT>::UType;
  const UTypeT* indices_to_sort = static_cast<const UTypeT*>(indices_before_sort);
  UTypeT* sorted_indice         = static_cast<UTypeT*>(indices_after_sort);
  void* cub_temp_storage        = nullptr;
  size_t temp_storage_bytes     = 0;
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  static_cast<IndexT*>(raw_indices),
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  static_cast<IndexT*>(raw_indices),
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  allocator.deallocate(reinterpret_cast<char*>(seq_indices),
                       wholememory_get_memory_size_from_array(&indices_desc));
  allocator.deallocate(static_cast<char*>(cub_temp_storage), temp_storage_bytes);
}

REGISTER_DISPATCH_ONE_TYPE(SortIndices, sort_indices_temp_func, SINT3264)

template <typename IndexT>
__global__ void mark_unique_indices_kernel(const IndexT* sorted_indices,
                                           int64_t indice_count,
                                           int64_t* head_flags)
{
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < indice_count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    head_flags[index] = index == 0 || sorted_indices[index] != sorted_indices[index - 1] ? 1 : 0;
  }
}

template <typename IndexT>
__global__ void compact_unique_indices_kernel(const IndexT* sorted_indices,
                                              const IndexT* raw_indices,
                                              int64_t indice_count,
                                              const int64_t* unique_prefix,
                                              IndexT* unique_indices,
                                              int64_t* inverse_indices)
{
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < indice_count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    auto const unique_index = unique_prefix[index] - 1;
    if (index == 0 || sorted_indices[index] != sorted_indices[index - 1]) {
      unique_indices[unique_index] = sorted_indices[index];
    }
    inverse_indices[static_cast<int64_t>(raw_indices[index])] = unique_index;
  }
}

template <typename IndexT>
void compact_sorted_unique_indices_temp_func(const void* sorted_indices,
                                             const void* raw_indices,
                                             wholememory_array_description_t indice_desc,
                                             void* unique_indices,
                                             int64_t* inverse_indices,
                                             int64_t* host_unique_count,
                                             wm_thrust_allocator* p_thrust_allocator,
                                             cudaStream_t stream)
{
  WHOLEMEMORY_CHECK(indice_desc.storage_offset == 0);
  WHOLEMEMORY_CHECK(host_unique_count != nullptr);
  *host_unique_count = 0;
  if (indice_desc.size == 0) { return; }

  auto& allocator          = *p_thrust_allocator;
  auto const prefix_bytes  = static_cast<size_t>(indice_desc.size) * sizeof(int64_t);
  auto* head_flags         = reinterpret_cast<int64_t*>(allocator.allocate(prefix_bytes));
  auto* unique_prefix      = reinterpret_cast<int64_t*>(allocator.allocate(prefix_bytes));
  constexpr int block_size = 256;
  auto const block_count   = static_cast<int>(std::min<int64_t>(
    (indice_desc.size + block_size - 1) / block_size, static_cast<int64_t>(65535)));

  mark_unique_indices_kernel<<<block_count, block_size, 0, stream>>>(
    static_cast<const IndexT*>(sorted_indices), indice_desc.size, head_flags);
  WM_CUDA_CHECK(cudaGetLastError());

  void* scan_storage = nullptr;
  size_t scan_bytes  = 0;
  cub::DeviceScan::InclusiveSum(
    scan_storage, scan_bytes, head_flags, unique_prefix, indice_desc.size, stream);
  scan_storage = allocator.allocate(scan_bytes);
  cub::DeviceScan::InclusiveSum(
    scan_storage, scan_bytes, head_flags, unique_prefix, indice_desc.size, stream);

  compact_unique_indices_kernel<<<block_count, block_size, 0, stream>>>(
    static_cast<const IndexT*>(sorted_indices),
    static_cast<const IndexT*>(raw_indices),
    indice_desc.size,
    unique_prefix,
    static_cast<IndexT*>(unique_indices),
    inverse_indices);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaMemcpyAsync(host_unique_count,
                                unique_prefix + indice_desc.size - 1,
                                sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));

  allocator.deallocate(static_cast<char*>(scan_storage), scan_bytes);
  allocator.deallocate(reinterpret_cast<char*>(unique_prefix), prefix_bytes);
  allocator.deallocate(reinterpret_cast<char*>(head_flags), prefix_bytes);
}

REGISTER_DISPATCH_ONE_TYPE(CompactSortedUniqueIndices,
                           compact_sorted_unique_indices_temp_func,
                           SINT3264)

wholememory_error_code_t sort_indices_func(const void* indices_before_sort,
                                           wholememory_array_description_t indice_desc,
                                           void* indices_after_sort,
                                           void* raw_indices,
                                           wm_thrust_allocator* p_thrust_allocator,
                                           wholememory_env_func_t* p_env_fns,
                                           cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortIndices,
                      indices_before_sort,
                      indice_desc,
                      indices_after_sort,
                      raw_indices,
                      p_thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("sort_indices_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("sort_indices_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t compact_sorted_unique_indices_func(
  const void* sorted_indices,
  const void* raw_indices,
  wholememory_array_description_t indice_desc,
  void* unique_indices,
  int64_t* inverse_indices,
  int64_t* host_unique_count,
  wm_thrust_allocator* p_thrust_allocator,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      CompactSortedUniqueIndices,
                      sorted_indices,
                      raw_indices,
                      indice_desc,
                      unique_indices,
                      inverse_indices,
                      host_unique_count,
                      p_thrust_allocator,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("compact_sorted_unique_indices_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("compact_sorted_unique_indices_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
