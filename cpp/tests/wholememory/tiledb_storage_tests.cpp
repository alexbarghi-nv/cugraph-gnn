/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "wholememory/tiledb_storage.hpp"

#include <gtest/gtest.h>
#include <tiledb/tiledb.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(int rc)
{
  if (rc != TILEDB_OK) { throw std::runtime_error("TileDB test setup failed"); }
}

class temporary_feature_array {
 public:
  temporary_feature_array()
  {
    uri_ = (std::filesystem::temp_directory_path() /
            ("wholememory_tiledb_storage_" + std::to_string(reinterpret_cast<uintptr_t>(this))))
             .string();

    tiledb_ctx_t* ctx = nullptr;
    check(tiledb_ctx_alloc(nullptr, &ctx));
    tiledb_dimension_t* dimension = nullptr;
    tiledb_domain_t* domain       = nullptr;
    tiledb_attribute_t* attribute = nullptr;
    tiledb_array_schema_t* schema = nullptr;
    std::array<int64_t, 2> row_domain{0, 4};
    int64_t tile_extent = 2;
    check(tiledb_dimension_alloc(
      ctx, "row", TILEDB_INT64, row_domain.data(), &tile_extent, &dimension));
    check(tiledb_domain_alloc(ctx, &domain));
    check(tiledb_domain_add_dimension(ctx, domain, dimension));
    check(tiledb_attribute_alloc(ctx, "values", TILEDB_UINT8, &attribute));
    check(tiledb_attribute_set_cell_val_num(ctx, attribute, sizeof(std::array<int32_t, 2>)));
    check(tiledb_array_schema_alloc(ctx, TILEDB_DENSE, &schema));
    check(tiledb_array_schema_set_domain(ctx, schema, domain));
    check(tiledb_array_schema_set_cell_order(ctx, schema, TILEDB_ROW_MAJOR));
    check(tiledb_array_schema_set_tile_order(ctx, schema, TILEDB_ROW_MAJOR));
    check(tiledb_array_schema_add_attribute(ctx, schema, attribute));
    check(tiledb_array_schema_check(ctx, schema));
    check(tiledb_array_create(ctx, uri_.c_str(), schema));

    tiledb_array_t* array = nullptr;
    tiledb_query_t* query = nullptr;
    check(tiledb_array_alloc(ctx, uri_.c_str(), &array));
    check(tiledb_array_open(ctx, array, TILEDB_WRITE));
    check(tiledb_query_alloc(ctx, array, TILEDB_WRITE, &query));
    check(tiledb_query_set_layout(ctx, query, TILEDB_ROW_MAJOR));
    std::array<std::array<int32_t, 2>, 5> values{
      {{{0, 1}}, {{10, 11}}, {{20, 21}}, {{30, 31}}, {{40, 41}}}};
    uint64_t values_size = sizeof(values);
    check(tiledb_query_set_data_buffer(ctx, query, "values", values.data(), &values_size));
    check(tiledb_query_submit(ctx, query));
    check(tiledb_array_close(ctx, array));

    tiledb_query_free(&query);
    tiledb_array_free(&array);
    tiledb_array_schema_free(&schema);
    tiledb_attribute_free(&attribute);
    tiledb_domain_free(&domain);
    tiledb_dimension_free(&dimension);
    tiledb_ctx_free(&ctx);
  }

  ~temporary_feature_array() { std::filesystem::remove_all(uri_); }
  [[nodiscard]] std::string const& uri() const { return uri_; }

 private:
  std::string uri_;
};

class scoped_query_chunk_rows {
 public:
  explicit scoped_query_chunk_rows(char const* value)
  {
    if (setenv("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS", value, 1) != 0) {
      throw std::runtime_error("could not set TileDB query chunk environment variable");
    }
  }
  ~scoped_query_chunk_rows() { unsetenv("WHOLEMEMORY_TILEDB_QUERY_CHUNK_ROWS"); }
};

TEST(TileDBStorage, PreservesRequestOrderDuplicatesAndColumnSlices)
{
  temporary_feature_array array;
  wholememory::tiledb_read_only_storage storage(array.uri(), sizeof(std::array<int32_t, 2>), 5);
  std::array<int64_t, 5> ids{4, 1, 4, 0, 2};
  std::vector<unsigned char> raw(ids.size() * storage.row_bytes());
  std::array<int32_t, 5> output{};

  auto const metrics = storage.read_rows(ids.data(),
                                         WHOLEMEMORY_DT_INT64,
                                         ids.size(),
                                         0,
                                         sizeof(int32_t),
                                         sizeof(int32_t),
                                         raw.data(),
                                         raw.size(),
                                         output.data());

  EXPECT_EQ(output, (std::array<int32_t, 5>{41, 11, 41, 1, 21}));
  EXPECT_GE(metrics.cpu_reorder_ms, 0.0);
  EXPECT_EQ(metrics.requested_rows, ids.size());
  EXPECT_EQ(metrics.unique_rows, size_t{4});
  EXPECT_EQ(metrics.range_count, size_t{2});
  EXPECT_EQ(metrics.query_count, size_t{1});
}

TEST(TileDBStorage, AcceptsGlobalIdsForALocalPartition)
{
  temporary_feature_array array;
  wholememory::tiledb_read_only_storage storage(array.uri(), sizeof(std::array<int32_t, 2>), 5);
  std::array<int32_t, 2> ids{102, 100};
  std::vector<unsigned char> raw(ids.size() * storage.row_bytes());
  std::array<std::array<int32_t, 2>, 2> output{};

  static_cast<void>(storage.read_rows(ids.data(),
                                      WHOLEMEMORY_DT_INT,
                                      ids.size(),
                                      100,
                                      0,
                                      storage.row_bytes(),
                                      raw.data(),
                                      raw.size(),
                                      output.data()));

  EXPECT_EQ(output[0], (std::array<int32_t, 2>{20, 21}));
  EXPECT_EQ(output[1], (std::array<int32_t, 2>{0, 1}));
}

TEST(TileDBStorage, BoundedQueriesPreserveOrderingAndDuplicates)
{
  temporary_feature_array array;
  scoped_query_chunk_rows chunk_rows("2");
  wholememory::tiledb_read_only_storage storage(array.uri(), sizeof(std::array<int32_t, 2>), 5);
  std::array<int64_t, 5> ids{4, 1, 4, 0, 2};
  std::vector<unsigned char> raw(ids.size() * storage.row_bytes());
  std::array<std::array<int32_t, 2>, 5> output{};

  static_cast<void>(storage.read_rows(ids.data(),
                                      WHOLEMEMORY_DT_INT64,
                                      ids.size(),
                                      0,
                                      0,
                                      storage.row_bytes(),
                                      raw.data(),
                                      raw.size(),
                                      output.data()));

  EXPECT_EQ(output[0], (std::array<int32_t, 2>{40, 41}));
  EXPECT_EQ(output[1], (std::array<int32_t, 2>{10, 11}));
  EXPECT_EQ(output[2], (std::array<int32_t, 2>{40, 41}));
  EXPECT_EQ(output[3], (std::array<int32_t, 2>{0, 1}));
  EXPECT_EQ(output[4], (std::array<int32_t, 2>{20, 21}));
}

}  // namespace
