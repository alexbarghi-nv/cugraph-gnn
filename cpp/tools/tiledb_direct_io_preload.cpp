// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

// Experimental LD_PRELOAD adapter for measuring TileDB local-file reads with O_DIRECT.
// This is benchmark tooling, not part of the WholeMemory runtime data path.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/stat.h>
#endif

namespace {

using open_fn  = int (*)(char const*, int, ...);
using openat_fn = int (*)(int, char const*, int, ...);
using pread_fn = ssize_t (*)(int, void*, size_t, off_t);
using pread64_fn = ssize_t (*)(int, void*, size_t, off64_t);

std::atomic<uint64_t> direct_open_attempts{0};
std::atomic<uint64_t> direct_open_successes{0};
std::atomic<uint64_t> direct_open_failures{0};
std::atomic<uint64_t> direct_pread_calls{0};
std::atomic<uint64_t> direct_requested_bytes{0};
std::atomic<uint64_t> direct_submitted_bytes{0};
std::atomic<uint64_t> direct_returned_bytes{0};
std::atomic<uint64_t> direct_pread_failures{0};

template <typename Function>
Function next_symbol(char const* name)
{
  auto* symbol = dlsym(RTLD_NEXT, name);
  if (symbol == nullptr) {
    errno = ENOSYS;
    return nullptr;
  }
  return reinterpret_cast<Function>(symbol);
}

bool direct_io_enabled()
{
  auto const* value = std::getenv("WHOLEMEMORY_TILEDB_DIRECT_IO");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool path_is_in_direct_io_root(char const* path)
{
  if (!direct_io_enabled() || path == nullptr || path[0] != '/') { return false; }
  auto const* root = std::getenv("WHOLEMEMORY_TILEDB_DIRECT_IO_ROOT");
  if (root == nullptr || root[0] != '/') { return false; }
  auto root_size = std::strlen(root);
  while (root_size > 1 && root[root_size - 1] == '/') { --root_size; }
  return std::strncmp(path, root, root_size) == 0 &&
         (path[root_size] == '\0' || path[root_size] == '/');
}

bool should_open_direct(char const* path, int flags)
{
  return (flags & O_ACCMODE) == O_RDONLY && path_is_in_direct_io_root(path);
}

bool flags_need_mode(int flags)
{
  return (flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE;
}

mode_t optional_mode(int flags, va_list arguments)
{
  return flags_need_mode(flags) ? static_cast<mode_t>(va_arg(arguments, int)) : 0;
}

int call_open(open_fn function, char const* path, int flags, mode_t mode)
{
  return flags_need_mode(flags) ? function(path, flags, mode) : function(path, flags);
}

int call_openat(openat_fn function, int dirfd, char const* path, int flags, mode_t mode)
{
  return flags_need_mode(flags) ? function(dirfd, path, flags, mode)
                                : function(dirfd, path, flags);
}

size_t round_up(size_t value, size_t alignment)
{
  if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    errno = EOVERFLOW;
    return 0;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

size_t power_of_two_at_least(size_t value)
{
  size_t result = sizeof(void*);
  while (result < value && result <= std::numeric_limits<size_t>::max() / 2) { result *= 2; }
  return result;
}

struct direct_alignment {
  size_t memory = 4096;
  size_t offset = 4096;
};

direct_alignment alignment_for_fd(int fd)
{
  direct_alignment result;
#if defined(__linux__) && defined(STATX_DIOALIGN)
  struct statx attributes {};
  if (statx(fd,
            "",
            AT_EMPTY_PATH | AT_STATX_DONT_SYNC,
            STATX_DIOALIGN,
            &attributes) == 0 &&
      (attributes.stx_mask & STATX_DIOALIGN) != 0) {
    if (attributes.stx_dio_mem_align != 0) { result.memory = attributes.stx_dio_mem_align; }
    if (attributes.stx_dio_offset_align != 0) { result.offset = attributes.stx_dio_offset_align; }
  }
#endif
  result.memory = power_of_two_at_least(result.memory);
  if (result.offset == 0) { result.offset = result.memory; }
  return result;
}

template <typename Offset, typename Function>
ssize_t direct_pread(int fd, void* buffer, size_t count, Offset offset, Function real_pread)
{
  if (count == 0) { return 0; }
  if (offset < 0) {
    errno = EINVAL;
    return -1;
  }
  auto const flags = fcntl(fd, F_GETFL);
  if (flags < 0 || (flags & O_DIRECT) == 0) { return real_pread(fd, buffer, count, offset); }

  direct_pread_calls.fetch_add(1, std::memory_order_relaxed);
  direct_requested_bytes.fetch_add(count, std::memory_order_relaxed);
  auto const alignment = alignment_for_fd(fd);
  auto const request_offset = static_cast<uint64_t>(offset);
  auto const aligned_offset = (request_offset / alignment.offset) * alignment.offset;
  auto const prefix = static_cast<size_t>(request_offset - aligned_offset);
  if (count > std::numeric_limits<size_t>::max() - prefix) {
    direct_pread_failures.fetch_add(1, std::memory_order_relaxed);
    errno = EOVERFLOW;
    return -1;
  }
  auto const aligned_count = round_up(prefix + count, alignment.offset);
  if (aligned_count == 0) {
    direct_pread_failures.fetch_add(1, std::memory_order_relaxed);
    return -1;
  }

  void* aligned_buffer = nullptr;
  auto const allocation_error = posix_memalign(&aligned_buffer, alignment.memory, aligned_count);
  if (allocation_error != 0) {
    direct_pread_failures.fetch_add(1, std::memory_order_relaxed);
    errno = allocation_error;
    return -1;
  }
  direct_submitted_bytes.fetch_add(aligned_count, std::memory_order_relaxed);
  auto const actual =
    real_pread(fd, aligned_buffer, aligned_count, static_cast<Offset>(aligned_offset));
  auto saved_errno = errno;
  if (actual < 0) {
    direct_pread_failures.fetch_add(1, std::memory_order_relaxed);
    std::free(aligned_buffer);
    errno = saved_errno;
    return -1;
  }
  auto const available = actual > static_cast<ssize_t>(prefix)
                           ? static_cast<size_t>(actual) - prefix
                           : 0;
  auto const returned = available < count ? available : count;
  if (returned != 0) {
    std::memcpy(buffer, static_cast<std::byte*>(aligned_buffer) + prefix, returned);
  }
  std::free(aligned_buffer);
  direct_returned_bytes.fetch_add(returned, std::memory_order_relaxed);
  errno = saved_errno;
  return static_cast<ssize_t>(returned);
}

}  // namespace

extern "C" int open(char const* path, int flags, ...)
{
  static auto real_open = next_symbol<open_fn>("open");
  if (real_open == nullptr) { return -1; }
  va_list arguments;
  va_start(arguments, flags);
  auto const mode = optional_mode(flags, arguments);
  va_end(arguments);
  if (!should_open_direct(path, flags)) { return call_open(real_open, path, flags, mode); }
  direct_open_attempts.fetch_add(1, std::memory_order_relaxed);
  auto const fd = call_open(real_open, path, flags | O_DIRECT, mode);
  (fd >= 0 ? direct_open_successes : direct_open_failures).fetch_add(
    1, std::memory_order_relaxed);
  return fd;
}

extern "C" int open64(char const* path, int flags, ...)
{
  static auto real_open = next_symbol<open_fn>("open64");
  if (real_open == nullptr) { return -1; }
  va_list arguments;
  va_start(arguments, flags);
  auto const mode = optional_mode(flags, arguments);
  va_end(arguments);
  if (!should_open_direct(path, flags)) { return call_open(real_open, path, flags, mode); }
  direct_open_attempts.fetch_add(1, std::memory_order_relaxed);
  auto const fd = call_open(real_open, path, flags | O_DIRECT, mode);
  (fd >= 0 ? direct_open_successes : direct_open_failures).fetch_add(
    1, std::memory_order_relaxed);
  return fd;
}

extern "C" int openat(int dirfd, char const* path, int flags, ...)
{
  static auto real_open = next_symbol<openat_fn>("openat");
  if (real_open == nullptr) { return -1; }
  va_list arguments;
  va_start(arguments, flags);
  auto const mode = optional_mode(flags, arguments);
  va_end(arguments);
  if (!should_open_direct(path, flags)) { return call_openat(real_open, dirfd, path, flags, mode); }
  direct_open_attempts.fetch_add(1, std::memory_order_relaxed);
  auto const fd = call_openat(real_open, dirfd, path, flags | O_DIRECT, mode);
  (fd >= 0 ? direct_open_successes : direct_open_failures).fetch_add(
    1, std::memory_order_relaxed);
  return fd;
}

extern "C" int openat64(int dirfd, char const* path, int flags, ...)
{
  static auto real_open = next_symbol<openat_fn>("openat64");
  if (real_open == nullptr) { return -1; }
  va_list arguments;
  va_start(arguments, flags);
  auto const mode = optional_mode(flags, arguments);
  va_end(arguments);
  if (!should_open_direct(path, flags)) { return call_openat(real_open, dirfd, path, flags, mode); }
  direct_open_attempts.fetch_add(1, std::memory_order_relaxed);
  auto const fd = call_openat(real_open, dirfd, path, flags | O_DIRECT, mode);
  (fd >= 0 ? direct_open_successes : direct_open_failures).fetch_add(
    1, std::memory_order_relaxed);
  return fd;
}

extern "C" ssize_t pread(int fd, void* buffer, size_t count, off_t offset)
{
  static auto real_pread = next_symbol<pread_fn>("pread");
  return real_pread == nullptr ? -1 : direct_pread(fd, buffer, count, offset, real_pread);
}

extern "C" ssize_t pread64(int fd, void* buffer, size_t count, off64_t offset)
{
  static auto real_pread = next_symbol<pread64_fn>("pread64");
  return real_pread == nullptr ? -1 : direct_pread(fd, buffer, count, offset, real_pread);
}

extern "C" uint32_t wholememory_tiledb_direct_io_preload_version() { return 1; }

extern "C" void wholememory_tiledb_direct_io_reset_counters()
{
  direct_open_attempts.store(0, std::memory_order_relaxed);
  direct_open_successes.store(0, std::memory_order_relaxed);
  direct_open_failures.store(0, std::memory_order_relaxed);
  direct_pread_calls.store(0, std::memory_order_relaxed);
  direct_requested_bytes.store(0, std::memory_order_relaxed);
  direct_submitted_bytes.store(0, std::memory_order_relaxed);
  direct_returned_bytes.store(0, std::memory_order_relaxed);
  direct_pread_failures.store(0, std::memory_order_relaxed);
}

extern "C" uint64_t wholememory_tiledb_direct_io_counter(uint32_t index)
{
  switch (index) {
    case 0: return direct_open_attempts.load(std::memory_order_relaxed);
    case 1: return direct_open_successes.load(std::memory_order_relaxed);
    case 2: return direct_open_failures.load(std::memory_order_relaxed);
    case 3: return direct_pread_calls.load(std::memory_order_relaxed);
    case 4: return direct_requested_bytes.load(std::memory_order_relaxed);
    case 5: return direct_submitted_bytes.load(std::memory_order_relaxed);
    case 6: return direct_returned_bytes.load(std::memory_order_relaxed);
    case 7: return direct_pread_failures.load(std::memory_order_relaxed);
    default: return 0;
  }
}
