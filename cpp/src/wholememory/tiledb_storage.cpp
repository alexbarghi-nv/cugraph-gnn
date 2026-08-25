/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tiledb_storage.hpp"

#include <tiledb/tiledb.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wholememory {
namespace {

void check_tiledb(int rc, tiledb_ctx_t* ctx, char const* operation)
{
  if (rc == TILEDB_OK) { return; }
  std::string message(operation);
  if (ctx != nullptr) {
    tiledb_error_t* error = nullptr;
    if (tiledb_ctx_get_last_error(ctx, &error) == TILEDB_OK && error != nullptr) {
      char const* detail = nullptr;
      if (tiledb_error_message(error, &detail) == TILEDB_OK && detail != nullptr) {
        message.append(": ").append(detail);
      }
      tiledb_error_free(&error);
    }
  }
  throw std::runtime_error(message);
}

int64_t get_id(const void* ids, wholememory_dtype_t dtype, size_t index)
{
  switch (dtype) {
    case WHOLEMEMORY_DT_INT: return static_cast<int64_t>(static_cast<int32_t const*>(ids)[index]);
    case WHOLEMEMORY_DT_INT64: return static_cast<int64_t const*>(ids)[index];
    default: throw std::invalid_argument("TileDB gather indices must be INT32 or INT64");
  }
}

size_t query_chunk_rows_from_environment()
{
  auto const* value = std::getenv("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) { return 0; }
  errno          = 0;
  char* end      = nullptr;
  auto const raw = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || raw == 0 ||
      raw > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(
      "WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS must be zero or a positive integer");
  }
  return static_cast<size_t>(raw);
}

}  // namespace

struct tiledb_read_only_storage::impl {
  impl(std::string array_uri, size_t bytes_per_row, int64_t rows)
    : uri(std::move(array_uri)),
      row_bytes(bytes_per_row),
      row_count(rows),
      query_chunk_rows(query_chunk_rows_from_environment())
  {
    if (uri.empty()) { throw std::invalid_argument("TileDB array URI must not be empty"); }
    if (row_bytes == 0) { throw std::invalid_argument("TileDB row size must be nonzero"); }
    if (row_count <= 0) { throw std::invalid_argument("TileDB row count must be positive"); }

    check_tiledb(tiledb_ctx_alloc(nullptr, &ctx), nullptr, "tiledb_ctx_alloc");
    try {
      tiledb_object_t object_type = TILEDB_INVALID;
      check_tiledb(tiledb_object_type(ctx, uri.c_str(), &object_type), ctx, "tiledb_object_type");
      if (object_type != TILEDB_ARRAY) {
        throw std::invalid_argument("TileDB URI is not an array: " + uri);
      }
      check_tiledb(tiledb_array_alloc(ctx, uri.c_str(), &array), ctx, "tiledb_array_alloc");
      check_tiledb(tiledb_array_open(ctx, array, TILEDB_READ), ctx, "tiledb_array_open");
    } catch (...) {
      if (array != nullptr) { tiledb_array_free(&array); }
      tiledb_ctx_free(&ctx);
      throw;
    }
  }

  ~impl()
  {
    if (array != nullptr) {
      tiledb_array_close(ctx, array);
      tiledb_array_free(&array);
    }
    if (ctx != nullptr) { tiledb_ctx_free(&ctx); }
  }

  std::string uri;
  size_t row_bytes;
  int64_t row_count;
  size_t query_chunk_rows;
  tiledb_ctx_t* ctx     = nullptr;
  tiledb_array_t* array = nullptr;
  mutable std::mutex query_mutex;
};

tiledb_read_only_storage::tiledb_read_only_storage(std::string uri,
                                                   size_t row_bytes,
                                                   int64_t row_count)
  : impl_(std::make_unique<impl>(std::move(uri), row_bytes, row_count))
{
}

tiledb_read_only_storage::~tiledb_read_only_storage() = default;

double tiledb_read_only_storage::read_rows(const void* ids,
                                           wholememory_dtype_t id_dtype,
                                           size_t id_count,
                                           int64_t global_row_offset,
                                           size_t column_byte_offset,
                                           size_t output_row_bytes,
                                           void* raw_rows,
                                           size_t raw_rows_size,
                                           void* output) const
{
  if (id_count == 0) { return 0.0; }
  if (ids == nullptr || raw_rows == nullptr || output == nullptr) {
    throw std::invalid_argument("TileDB read buffers must not be null");
  }
  if (column_byte_offset > impl_->row_bytes ||
      output_row_bytes > impl_->row_bytes - column_byte_offset) {
    throw std::out_of_range("TileDB gather column slice is outside the stored row");
  }
  if (id_count > std::numeric_limits<size_t>::max() / impl_->row_bytes ||
      raw_rows_size < id_count * impl_->row_bytes) {
    throw std::invalid_argument("TileDB raw read buffer is too small");
  }

  struct indexed_id {
    int64_t local_id;
    size_t original_position;
  };
  std::vector<indexed_id> sorted_ids;
  sorted_ids.reserve(id_count);
  for (size_t i = 0; i < id_count; ++i) {
    auto const global_id = get_id(ids, id_dtype, i);
    auto const local_id  = global_id - global_row_offset;
    if (local_id < 0 || local_id >= impl_->row_count) {
      throw std::out_of_range("TileDB gather row is outside the rank-local partition");
    }
    sorted_ids.push_back({local_id, i});
  }
  std::sort(sorted_ids.begin(), sorted_ids.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.local_id < rhs.local_id ||
           (lhs.local_id == rhs.local_id && lhs.original_position < rhs.original_position);
  });

  std::vector<int64_t> unique_ids;
  unique_ids.reserve(sorted_ids.size());
  for (auto const& item : sorted_ids) {
    if (unique_ids.empty() || item.local_id != unique_ids.back()) {
      unique_ids.push_back(item.local_id);
    }
  }

  // TileDB contexts can service concurrent work, but the array object is shared by this handle.
  // Serialize query setup/submission until per-thread array handles are justified by measurements.
  std::scoped_lock query_lock(impl_->query_mutex);
  auto const query_chunk_rows = impl_->query_chunk_rows == 0
                                  ? unique_ids.size()
                                  : std::min(impl_->query_chunk_rows, unique_ids.size());
  for (size_t chunk_begin = 0; chunk_begin < unique_ids.size(); chunk_begin += query_chunk_rows) {
    auto const chunk_end        = std::min(chunk_begin + query_chunk_rows, unique_ids.size());
    tiledb_query_t* query       = nullptr;
    tiledb_subarray_t* subarray = nullptr;
    try {
      check_tiledb(tiledb_query_alloc(impl_->ctx, impl_->array, TILEDB_READ, &query),
                   impl_->ctx,
                   "query alloc");
      check_tiledb(
        tiledb_query_set_layout(impl_->ctx, query, TILEDB_ROW_MAJOR), impl_->ctx, "query layout");
      check_tiledb(
        tiledb_subarray_alloc(impl_->ctx, impl_->array, &subarray), impl_->ctx, "subarray alloc");

      // Coalesce adjacent point ids. TileDB operates on tiles, so ranges avoid needless query-range
      // metadata while still returning rows in ascending order.
      for (size_t begin = chunk_begin; begin < chunk_end;) {
        size_t end = begin;
        while (end + 1 < chunk_end && unique_ids[end + 1] == unique_ids[end] + 1) {
          ++end;
        }
        auto const range_start = unique_ids[begin];
        auto const range_end   = unique_ids[end];
        check_tiledb(
          tiledb_subarray_add_range(impl_->ctx, subarray, 0, &range_start, &range_end, nullptr),
          impl_->ctx,
          "add row range");
        begin = end + 1;
      }

      check_tiledb(
        tiledb_query_set_subarray_t(impl_->ctx, query, subarray), impl_->ctx, "set subarray");
      auto* chunk_output = static_cast<unsigned char*>(raw_rows) + chunk_begin * impl_->row_bytes;
      uint64_t result_bytes = (chunk_end - chunk_begin) * impl_->row_bytes;
      check_tiledb(
        tiledb_query_set_data_buffer(impl_->ctx, query, "values", chunk_output, &result_bytes),
        impl_->ctx,
        "set values buffer");
      check_tiledb(tiledb_query_submit(impl_->ctx, query), impl_->ctx, "query submit");

      tiledb_query_status_t status = TILEDB_UNINITIALIZED;
      check_tiledb(tiledb_query_get_status(impl_->ctx, query, &status), impl_->ctx, "query status");
      auto const expected_bytes = (chunk_end - chunk_begin) * impl_->row_bytes;
      if (status != TILEDB_COMPLETED || result_bytes != expected_bytes) {
        throw std::runtime_error("TileDB query did not return every requested feature row");
      }
    } catch (...) {
      if (subarray != nullptr) { tiledb_subarray_free(&subarray); }
      if (query != nullptr) { tiledb_query_free(&query); }
      throw;
    }
    tiledb_subarray_free(&subarray);
    tiledb_query_free(&query);
  }

  auto const reorder_start = std::chrono::steady_clock::now();
  auto const* raw          = static_cast<unsigned char const*>(raw_rows);
  auto* dst                = static_cast<unsigned char*>(output);
  size_t unique_position   = 0;
  for (size_t i = 0; i < sorted_ids.size();) {
    size_t end = i + 1;
    while (end < sorted_ids.size() && sorted_ids[end].local_id == sorted_ids[i].local_id) {
      ++end;
    }
    auto const* source = raw + unique_position * impl_->row_bytes + column_byte_offset;
    for (size_t j = i; j < end; ++j) {
      std::memcpy(
        dst + sorted_ids[j].original_position * output_row_bytes, source, output_row_bytes);
    }
    ++unique_position;
    i = end;
  }
  auto const reorder_elapsed = std::chrono::steady_clock::now() - reorder_start;
  return std::chrono::duration<double, std::milli>(reorder_elapsed).count();
}

std::string const& tiledb_read_only_storage::uri() const noexcept { return impl_->uri; }
size_t tiledb_read_only_storage::row_bytes() const noexcept { return impl_->row_bytes; }
int64_t tiledb_read_only_storage::row_count() const noexcept { return impl_->row_count; }

}  // namespace wholememory
