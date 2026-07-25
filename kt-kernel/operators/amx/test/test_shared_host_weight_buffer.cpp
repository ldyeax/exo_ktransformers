#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "../la/amx.hpp"

int main() {
  using Kernel = amx::GemmKernel224Int4;
  using Buffer = Kernel::BufferB;
  constexpr int m = 32;
  constexpr int n = 32;
  constexpr int k = 128;
  const size_t packed_bytes = sizeof(int8_t) * n * k / 2;
  const size_t scale_bytes = sizeof(float) * n;
  const size_t private_bytes = Buffer::required_size(n, k);

  auto* private_storage = std::aligned_alloc(64, private_bytes);
  assert(private_storage != nullptr);
  auto copying_buffer = std::make_shared<Buffer>(n, k, private_storage);
  std::memset(copying_buffer->b, 0x21, packed_bytes);
  for (int index = 0; index < n; ++index) {
    copying_buffer->d[index] = 0.5f + static_cast<float>(index) / 64.0f;
  }

  const long page_size_result = sysconf(_SC_PAGESIZE);
  assert(page_size_result > 0);
  const size_t page_size = static_cast<size_t>(page_size_result);
  auto* weight_mapping =
      static_cast<uint8_t*>(mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  auto* scale_mapping =
      static_cast<uint8_t*>(mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(weight_mapping != MAP_FAILED && scale_mapping != MAP_FAILED);

  // safetensors promises 8-byte offsets, not the legacy BufferB allocator's
  // 64-byte alignment. Exercise both AMX and AVX paths at exactly 8 mod 64.
  auto* source_weight = weight_mapping + 8;
  auto* source_scale = reinterpret_cast<float*>(scale_mapping + 8);
  assert(reinterpret_cast<uintptr_t>(source_weight) % 64 == 8);
  assert(reinterpret_cast<uintptr_t>(source_scale) % 64 == 8);
  std::memcpy(source_weight, copying_buffer->b, packed_bytes);
  std::memcpy(source_scale, copying_buffer->d, scale_bytes);

  Buffer alignment_guard(n, k, nullptr);
  bool rejected_four_byte_weight = false;
  try {
    alignment_guard.set_external_readonly_data(weight_mapping + 4, source_scale);
  } catch (const std::runtime_error&) {
    rejected_four_byte_weight = true;
  }
  assert(rejected_four_byte_weight);

  auto shared_buffer = std::make_shared<Buffer>(n, k, nullptr);
  shared_buffer->set_external_readonly_data(source_weight, source_scale);
  assert(shared_buffer->external_readonly);
  assert(shared_buffer->b == source_weight);
  assert(shared_buffer->d == source_scale);
  assert(copying_buffer->b != source_weight);
  assert(copying_buffer->d != source_scale);

  assert(mprotect(weight_mapping, page_size, PROT_READ) == 0);
  assert(mprotect(scale_mapping, page_size, PROT_READ) == 0);

  const size_t input_bytes = Kernel::BufferA::required_size(m, k);
  const size_t output_bytes = Kernel::BufferC::required_size(m, n);
  void* input_storage = std::aligned_alloc(64, input_bytes);
  void* copying_output_storage = std::aligned_alloc(64, output_bytes);
  void* shared_output_storage = std::aligned_alloc(64, output_bytes);
  assert(input_storage != nullptr && copying_output_storage != nullptr && shared_output_storage != nullptr);
  auto input = std::make_shared<Kernel::BufferA>(m, k, input_storage);
  auto copying_output = std::make_shared<Kernel::BufferC>(m, n, copying_output_storage);
  auto shared_output = std::make_shared<Kernel::BufferC>(m, n, shared_output_storage);
  std::memset(input->a, 1, static_cast<size_t>(m) * k);
  for (int index = 0; index < m; ++index) {
    input->d[index] = 0.25f + static_cast<float>(index) / 64.0f;
  }

  std::memset(copying_output_storage, 0, output_bytes);
  std::memset(shared_output_storage, 0, output_bytes);
  amx::vec_mul(m, n, k, input, copying_buffer, copying_output, 0, 1);
  amx::vec_mul(m, n, k, input, shared_buffer, shared_output, 0, 1);
  assert(std::memcmp(copying_output_storage, shared_output_storage, output_bytes) == 0);

  Kernel::config();
  std::memset(copying_output_storage, 0, output_bytes);
  std::memset(shared_output_storage, 0, output_bytes);
  amx::mat_mul(m, n, k, input, copying_buffer, copying_output, 0, 1);
  amx::mat_mul(m, n, k, input, shared_buffer, shared_output, 0, 1);
  assert(std::memcmp(copying_output_storage, shared_output_storage, output_bytes) == 0);

  // Destruction only drops the view. It must not free or unmap the external
  // mappings, which remain readable until their actual owner releases them.
  shared_buffer.reset();
  assert(std::memcmp(source_weight, copying_buffer->b, packed_bytes) == 0);
  assert(std::memcmp(source_scale, copying_buffer->d, scale_bytes) == 0);

  std::free(shared_output_storage);
  std::free(copying_output_storage);
  std::free(input_storage);
  std::free(private_storage);
  assert(munmap(scale_mapping, page_size) == 0);
  assert(munmap(weight_mapping, page_size) == 0);
  return 0;
}
