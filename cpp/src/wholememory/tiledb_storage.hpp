/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/tensor_description.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace wholememory {

struct tiledb_read_metrics {
  double id_decode_ms{};
  double id_sort_ms{};
  double id_deduplicate_ms{};
  double query_setup_ms{};
  double range_build_ms{};
  double query_submit_ms{};
  double cpu_reorder_ms{};
  size_t requested_rows{};
  size_t unique_rows{};
  size_t range_count{};
  size_t query_count{};
};

/**
 * Read-only feature storage backed by a one-dimensional dense TileDB array.
 *
 * The array must have an INT64 dimension named "row" whose domain starts at zero and a
 * fixed-sized UINT8 attribute named "values" with cell_val_num equal to row_bytes.
 */
class tiledb_read_only_storage {
 public:
  tiledb_read_only_storage(std::string uri, size_t row_bytes, int64_t row_count);
  ~tiledb_read_only_storage();

  tiledb_read_only_storage(tiledb_read_only_storage const&)            = delete;
  tiledb_read_only_storage& operator=(tiledb_read_only_storage const&) = delete;

  /**
   * Read global row ids and restore their original order into output.
   *
   * raw_rows is caller-owned scratch storage of at least id_count * row_bytes bytes. Keeping this
   * allocation outside TileDB allows WholeMemory to supply CUDA-pinned memory.
   */
  [[nodiscard]] tiledb_read_metrics read_rows(const void* ids,
                                              wholememory_dtype_t id_dtype,
                                              size_t id_count,
                                              int64_t global_row_offset,
                                              size_t column_byte_offset,
                                              size_t output_row_bytes,
                                              void* raw_rows,
                                              size_t raw_rows_size,
                                              void* output) const;

  [[nodiscard]] std::string const& uri() const noexcept;
  [[nodiscard]] size_t row_bytes() const noexcept;
  [[nodiscard]] int64_t row_count() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace wholememory
