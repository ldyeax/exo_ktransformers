#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../operators/amx/fp4-moe.hpp"

namespace {

using Kernel = amx::GemmKernel224MXFP4SmallKGroup;

void* allocate_aligned(size_t bytes) {
  const size_t rounded = (bytes + 63) & ~size_t{63};
  void* storage = std::aligned_alloc(64, rounded);
  if (storage == nullptr) std::abort();
  std::memset(storage, 0, rounded);
  return storage;
}

ggml_bf16_t to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return ggml_bf16_t{static_cast<uint16_t>(bits >> 16)};
}

template <typename Function>
double minimum_seconds(int iterations, Function&& function) {
  double best = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(end - start).count());
  }
  return best;
}

void benchmark_shape(int m, int k) {
  constexpr int n = 256;
  constexpr int group_size = 32;
  void* source_storage =
      allocate_aligned(static_cast<size_t>(m) * k * sizeof(ggml_bf16_t));
  void* input_storage =
      allocate_aligned(Kernel::BufferA::required_size(m, k));
  void* weight_storage =
      allocate_aligned(Kernel::BufferB::required_size(n, k, group_size));
  void* avx_output_storage =
      allocate_aligned(Kernel::BufferC::required_size(m, n));
  void* amx_output_storage =
      allocate_aligned(Kernel::BufferC::required_size(m, n));

  auto* source = static_cast<ggml_bf16_t*>(source_storage);
  for (int index = 0; index < m * k; ++index) {
    source[index] = to_bf16(static_cast<float>((index % 31) - 15) / 256.0f);
  }
  auto input = std::make_shared<Kernel::BufferA>(m, k, input_storage);
  input->from_mat(m, source, 0, 1);

  auto weights =
      std::make_shared<Kernel::BufferB>(n, k, group_size, weight_storage);
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2);
  for (size_t index = 0; index < packed.size(); ++index) {
    packed[index] = static_cast<uint8_t>((index * 37 + 0x52) & 0xff);
  }
  weights->from_raw_mat(packed.data(), 0, 1);
  for (int row = 0; row < n; ++row) {
    for (int group = 0; group < k / group_size; ++group) {
      weights->d[row * (k / group_size) + group] =
          static_cast<uint8_t>(122 + ((row + group) % 4));
    }
  }

  auto avx_output =
      std::make_shared<Kernel::BufferC>(m, n, avx_output_storage);
  auto amx_output =
      std::make_shared<Kernel::BufferC>(m, n, amx_output_storage);
  Kernel::config();
  Kernel::fp4_mat_mat_kgroup(m, n, k, group_size, input.get(), weights.get(),
                             avx_output.get(), 0, 1);
  Kernel::fp4_mat_mat_amx_kgroup(m, n, k, group_size, input.get(),
                                 weights.get(), amx_output.get(), 0, 1);

  const double avx_seconds = minimum_seconds(3, [&] {
    Kernel::fp4_mat_mat_kgroup(m, n, k, group_size, input.get(), weights.get(),
                               avx_output.get(), 0, 1);
  });
  const double amx_seconds = minimum_seconds(5, [&] {
    Kernel::fp4_mat_mat_amx_kgroup(m, n, k, group_size, input.get(),
                                   weights.get(), amx_output.get(), 0, 1);
  });
  const double operations = 2.0 * m * n * k;
  std::printf(
      "m=%4d k=%4d avx_ms=%9.3f amx_ms=%9.3f speedup=%6.3fx "
      "avx_gflops=%8.2f amx_gflops=%8.2f\n",
      m, k, avx_seconds * 1000.0, amx_seconds * 1000.0,
      avx_seconds / amx_seconds, operations / avx_seconds / 1.0e9,
      operations / amx_seconds / 1.0e9);

  constexpr int decode_repetitions = 10;
  Kernel::fp4_mat_vec_kgroup(m, n, k, group_size, input.get(), weights.get(),
                             avx_output.get(), 0, 1);
  const double decode_seconds = minimum_seconds(7, [&] {
    for (int repetition = 0; repetition < decode_repetitions; ++repetition) {
      Kernel::fp4_mat_vec_kgroup(m, n, k, group_size, input.get(),
                                 weights.get(), avx_output.get(), 0, 1);
    }
  }) / decode_repetitions;
  std::printf("decode m=%4d k=%4d avx_ms=%9.3f avx_gflops=%8.2f\n", m, k,
              decode_seconds * 1000.0, operations / decode_seconds / 1.0e9);

  std::free(amx_output_storage);
  std::free(avx_output_storage);
  std::free(weight_storage);
  std::free(input_storage);
  std::free(source_storage);
}

}  // namespace

int main() {
  for (const int m : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    benchmark_shape(m, 4096);
  }
  for (const int m : {1, 2, 4, 6, 8, 32, 128}) {
    benchmark_shape(m, 7168);
  }
  return 0;
}
