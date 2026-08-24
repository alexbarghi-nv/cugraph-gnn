/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tiledb/tiledb.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(int rc, tiledb_ctx_t* ctx, char const* operation)
{
  if (rc == TILEDB_OK) { return; }
  std::string message(operation);
  tiledb_error_t* error = nullptr;
  if (ctx != nullptr && tiledb_ctx_get_last_error(ctx, &error) == TILEDB_OK && error != nullptr) {
    char const* detail = nullptr;
    if (tiledb_error_message(error, &detail) == TILEDB_OK && detail != nullptr) {
      message.append(": ").append(detail);
    }
    tiledb_error_free(&error);
  }
  throw std::runtime_error(message);
}

uint64_t parse_positive(char const* value, char const* name)
{
  size_t parsed = 0;
  auto result   = std::stoull(value, &parsed);
  if (value[parsed] != '\0' || result == 0) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return result;
}

bool parse_boolean(char const* value, char const* name)
{
  std::string const parsed(value);
  if (parsed == "0" || parsed == "false" || parsed == "off") { return false; }
  if (parsed == "1" || parsed == "true" || parsed == "on") { return true; }
  throw std::invalid_argument(std::string(name) + " must be 0/1, false/true, or off/on");
}

void create_array(tiledb_ctx_t* ctx,
                  std::string const& uri,
                  int64_t row_count,
                  uint32_t row_bytes,
                  int64_t tile_rows)
{
  tiledb_dimension_t* dimension = nullptr;
  tiledb_domain_t* domain       = nullptr;
  tiledb_attribute_t* attribute = nullptr;
  tiledb_array_schema_t* schema = nullptr;
  int64_t row_domain[2]         = {0, row_count - 1};
  try {
    check(tiledb_dimension_alloc(ctx, "row", TILEDB_INT64, row_domain, &tile_rows, &dimension),
          ctx,
          "create row dimension");
    check(tiledb_domain_alloc(ctx, &domain), ctx, "create domain");
    check(tiledb_domain_add_dimension(ctx, domain, dimension), ctx, "add row dimension");
    check(tiledb_attribute_alloc(ctx, "values", TILEDB_UINT8, &attribute),
          ctx,
          "create values attribute");
    check(tiledb_attribute_set_cell_val_num(ctx, attribute, row_bytes), ctx, "set values width");
    check(tiledb_array_schema_alloc(ctx, TILEDB_DENSE, &schema), ctx, "create schema");
    check(tiledb_array_schema_set_domain(ctx, schema, domain), ctx, "set domain");
    check(tiledb_array_schema_set_cell_order(ctx, schema, TILEDB_ROW_MAJOR), ctx, "set cell order");
    check(tiledb_array_schema_set_tile_order(ctx, schema, TILEDB_ROW_MAJOR), ctx, "set tile order");
    check(tiledb_array_schema_add_attribute(ctx, schema, attribute), ctx, "add values attribute");
    check(tiledb_array_schema_check(ctx, schema), ctx, "validate schema");
    check(tiledb_array_create(ctx, uri.c_str(), schema), ctx, "create array");
  } catch (...) {
    tiledb_array_schema_free(&schema);
    tiledb_attribute_free(&attribute);
    tiledb_domain_free(&domain);
    tiledb_dimension_free(&dimension);
    throw;
  }
  tiledb_array_schema_free(&schema);
  tiledb_attribute_free(&attribute);
  tiledb_domain_free(&domain);
  tiledb_dimension_free(&dimension);
}

void ingest(tiledb_ctx_t* ctx,
            std::string const& uri,
            std::filesystem::path const& input_path,
            int64_t row_count,
            size_t row_bytes,
            size_t chunk_rows)
{
  std::ifstream input(input_path, std::ios::binary);
  if (!input) { throw std::runtime_error("could not open input file: " + input_path.string()); }

  tiledb_array_t* array = nullptr;
  check(tiledb_array_alloc(ctx, uri.c_str(), &array), ctx, "allocate output array");
  try {
    check(tiledb_array_open(ctx, array, TILEDB_WRITE), ctx, "open output array");
    std::vector<unsigned char> buffer(chunk_rows * row_bytes);
    for (int64_t row_start = 0; row_start < row_count;) {
      auto const rows  = std::min<size_t>(chunk_rows, static_cast<size_t>(row_count - row_start));
      auto const bytes = rows * row_bytes;
      input.read(reinterpret_cast<char*>(buffer.data()), bytes);
      if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("input file ended before all rows were read");
      }

      tiledb_query_t* query       = nullptr;
      tiledb_subarray_t* subarray = nullptr;
      try {
        check(tiledb_query_alloc(ctx, array, TILEDB_WRITE, &query), ctx, "allocate write query");
        check(tiledb_query_set_layout(ctx, query, TILEDB_ROW_MAJOR), ctx, "set write layout");
        check(tiledb_subarray_alloc(ctx, array, &subarray), ctx, "allocate write subarray");
        int64_t row_end = row_start + static_cast<int64_t>(rows) - 1;
        check(tiledb_subarray_add_range(ctx, subarray, 0, &row_start, &row_end, nullptr),
              ctx,
              "set write row range");
        check(tiledb_query_set_subarray_t(ctx, query, subarray), ctx, "set write subarray");
        uint64_t buffer_bytes = bytes;
        check(tiledb_query_set_data_buffer(ctx, query, "values", buffer.data(), &buffer_bytes),
              ctx,
              "set write buffer");
        check(tiledb_query_submit(ctx, query), ctx, "submit write query");
      } catch (...) {
        tiledb_subarray_free(&subarray);
        tiledb_query_free(&query);
        throw;
      }
      tiledb_subarray_free(&subarray);
      tiledb_query_free(&query);
      row_start += rows;
    }
    check(tiledb_array_close(ctx, array), ctx, "close output array");
  } catch (...) {
    tiledb_array_close(ctx, array);
    tiledb_array_free(&array);
    throw;
  }
  tiledb_array_free(&array);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 5 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " ARRAY_URI RAW_FILE ROW_COUNT ROW_BYTES [TILE_ROWS=4096] [CHUNK_ROWS=1048576]"
                 " [CONSOLIDATE=0]\n";
    return 2;
  }
  try {
    std::string const uri       = argv[1];
    std::filesystem::path input = argv[2];
    auto const row_count_u64    = parse_positive(argv[3], "ROW_COUNT");
    auto const row_bytes_u64    = parse_positive(argv[4], "ROW_BYTES");
    auto const tile_rows_u64    = argc >= 6 ? parse_positive(argv[5], "TILE_ROWS") : 4096;
    auto const chunk_rows_u64   = argc >= 7 ? parse_positive(argv[6], "CHUNK_ROWS") : 1048576;
    auto const consolidate      = argc >= 8 ? parse_boolean(argv[7], "CONSOLIDATE") : false;
    if (row_count_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        row_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        tile_rows_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        chunk_rows_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        row_bytes_u64 > std::numeric_limits<uint64_t>::max() / row_count_u64 ||
        chunk_rows_u64 > std::numeric_limits<size_t>::max() / row_bytes_u64 ||
        chunk_rows_u64 * row_bytes_u64 >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::out_of_range("numeric argument is too large");
    }
    auto const expected_bytes = row_count_u64 * row_bytes_u64;
    if (!std::filesystem::is_regular_file(input) ||
        std::filesystem::file_size(input) != expected_bytes) {
      throw std::invalid_argument("RAW_FILE size must equal ROW_COUNT * ROW_BYTES");
    }

    tiledb_ctx_t* ctx = nullptr;
    check(tiledb_ctx_alloc(nullptr, &ctx), nullptr, "create TileDB context");
    try {
      tiledb_object_t existing = TILEDB_INVALID;
      check(tiledb_object_type(ctx, uri.c_str(), &existing), ctx, "check output URI");
      if (existing != TILEDB_INVALID) { throw std::invalid_argument("ARRAY_URI already exists"); }
      create_array(ctx,
                   uri,
                   static_cast<int64_t>(row_count_u64),
                   static_cast<uint32_t>(row_bytes_u64),
                   static_cast<int64_t>(tile_rows_u64));
      ingest(ctx,
             uri,
             input,
             static_cast<int64_t>(row_count_u64),
             static_cast<size_t>(row_bytes_u64),
             static_cast<size_t>(chunk_rows_u64));
      if (consolidate) {
        check(tiledb_array_consolidate(ctx, uri.c_str(), nullptr), ctx, "consolidate array");
        check(tiledb_array_vacuum(ctx, uri.c_str(), nullptr), ctx, "vacuum array");
      }
    } catch (...) {
      tiledb_ctx_free(&ctx);
      throw;
    }
    tiledb_ctx_free(&ctx);
    std::cout << "Created " << uri << " with " << row_count_u64 << " rows of " << row_bytes_u64
              << " bytes" << (consolidate ? " (consolidated)" : "") << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "wholememory_tiledb_ingest: " << error.what() << '\n';
    return 1;
  }
}
