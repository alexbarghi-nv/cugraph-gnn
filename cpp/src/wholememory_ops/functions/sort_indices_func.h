/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

#include <cstdint>

namespace wholememory_ops {

wholememory_error_code_t sort_indices_func(const void* indices_before_sort,
                                           wholememory_array_description_t indice_desc,
                                           void* indices_after_sort,
                                           void* raw_indices,
                                           wm_thrust_allocator* p_thrust_allocator,
                                           wholememory_env_func_t* p_env_fns,
                                           cudaStream_t stream);

/**
 * Compact sorted indices and build a mapping from each original position to its unique row.
 *
 * raw_indices must contain the sorted-position to original-position permutation produced by
 * sort_indices_func. unique_indices has capacity for indice_desc.size entries and inverse_indices
 * has exactly indice_desc.size INT64 entries. This function synchronizes stream before returning
 * because the host unique count is needed to size the TileDB CPU query.
 */
wholememory_error_code_t compact_sorted_unique_indices_func(
  const void* sorted_indices,
  const void* raw_indices,
  wholememory_array_description_t indice_desc,
  void* unique_indices,
  int64_t* inverse_indices,
  int64_t* host_unique_count,
  wm_thrust_allocator* p_thrust_allocator,
  cudaStream_t stream);

}  // namespace wholememory_ops
