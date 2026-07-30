#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "../fp4-moe.hpp"

namespace {

using Kernel = amx::GemmKernel224MXFP4SmallKGroup;

void* allocate_aligned(size_t bytes) {
  const size_t rounded = (bytes + 63) & ~size_t{63};
  void* storage = std::aligned_alloc(64, rounded);
  assert(storage != nullptr);
  std::memset(storage, 0, rounded);
  return storage;
}

ggml_bf16_t to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return ggml_bf16_t{static_cast<uint16_t>(bits >> 16)};
}

std::vector<float> logical_output(Kernel::BufferC* output, int m, int n) {
  std::vector<float> values(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    const float* source = output->get_submat(m, n, row, 0);
    std::memcpy(values.data() + static_cast<size_t>(row) * n, source,
                static_cast<size_t>(n) * sizeof(float));
  }
  return values;
}

void check_decoder() {
  alignas(16) uint8_t packed_bytes[16];
  alignas(64) uint16_t expected[32];
  alignas(64) uint16_t actual[32];
  std::mt19937 generator(0x56504552u);
  std::uniform_int_distribution<int> byte_distribution(0, 255);

  for (int trial = 0; trial < 4096; ++trial) {
    for (uint8_t& value : packed_bytes) {
      value = static_cast<uint8_t>(byte_distribution(generator));
    }
    const __m128i packed =
        _mm_load_si128(reinterpret_cast<const __m128i*>(packed_bytes));
    _mm512_store_si512(
        reinterpret_cast<__m512i*>(expected),
        Kernel::mxfp4_to_bf16_32_reference(packed));
    _mm512_store_si512(reinterpret_cast<__m512i*>(actual),
                       Kernel::mxfp4_to_bf16_32(packed));
    assert(std::memcmp(expected, actual, sizeof(expected)) == 0);
  }
}

void check_shape(int m) {
  constexpr int n = 64;
  constexpr int k = 128;
  constexpr int group_size = 32;
  std::mt19937 generator(0x4d584650u + static_cast<uint32_t>(m));
  std::uniform_int_distribution<int> nibble_distribution(0, 15);
  std::uniform_real_distribution<float> activation_distribution(-0.125f, 0.125f);
  // Native UE8M0 exponents for {2^-5, 2^-4, 2^-3, 2^-2}.
  constexpr uint8_t scale_exponents[] = {122, 123, 124, 125};

  const size_t input_bytes = Kernel::BufferA::required_size(m, k);
  const size_t weight_bytes = Kernel::BufferB::required_size(n, k, group_size);
  const size_t output_bytes = Kernel::BufferC::required_size(m, n);
  void* input_storage = allocate_aligned(input_bytes);
  void* weight_storage = allocate_aligned(weight_bytes);
  void* reference_storage = allocate_aligned(output_bytes);
  void* amx_storage = allocate_aligned(output_bytes);
  void* decode_storage = allocate_aligned(output_bytes);
  void* source_input_storage =
      allocate_aligned(static_cast<size_t>(m) * k * sizeof(ggml_bf16_t));

  auto input = std::make_shared<Kernel::BufferA>(m, k, input_storage);
  auto weights =
      std::make_shared<Kernel::BufferB>(n, k, group_size, weight_storage);
  auto reference =
      std::make_shared<Kernel::BufferC>(m, n, reference_storage);
  auto amx_output = std::make_shared<Kernel::BufferC>(m, n, amx_storage);
  auto decode_output =
      std::make_shared<Kernel::BufferC>(m, n, decode_storage);

  auto* source_input = static_cast<ggml_bf16_t*>(source_input_storage);
  for (int index = 0; index < m * k; ++index) {
    source_input[index] = to_bf16(activation_distribution(generator));
  }
  input->from_mat(m, source_input, 0, 1);

  std::vector<uint8_t> packed_weights(static_cast<size_t>(n) * k / 2);
  for (uint8_t& packed : packed_weights) {
    const uint8_t low = static_cast<uint8_t>(nibble_distribution(generator));
    const uint8_t high = static_cast<uint8_t>(nibble_distribution(generator));
    packed = static_cast<uint8_t>(low | (high << 4));
  }
  weights->from_raw_mat(packed_weights.data(), 0, 1);
  for (int row = 0; row < n; ++row) {
    for (int group = 0; group < k / group_size; ++group) {
      weights->d[row * (k / group_size) + group] =
          scale_exponents[(row + group) % std::size(scale_exponents)];
    }
  }

  Kernel::fp4_mat_mat_kgroup(m, n, k, group_size, input.get(), weights.get(),
                             reference.get(), 0, 1);
  Kernel::fp4_mat_vec_kgroup(m, n, k, group_size, input.get(), weights.get(),
                             decode_output.get(), 0, 1);
  Kernel::config();
  Kernel::fp4_mat_mat_amx_kgroup(m, n, k, group_size, input.get(),
                                 weights.get(), amx_output.get(), 0, 1);

  const std::vector<float> expected = logical_output(reference.get(), m, n);
  const std::vector<float> actual = logical_output(amx_output.get(), m, n);
  const std::vector<float> decoded =
      logical_output(decode_output.get(), m, n);
  assert(decoded == expected);
  double squared_error = 0.0;
  double squared_reference = 0.0;
  float maximum_absolute_error = 0.0f;
  for (size_t index = 0; index < expected.size(); ++index) {
    const float difference = actual[index] - expected[index];
    squared_error += static_cast<double>(difference) * difference;
    squared_reference +=
        static_cast<double>(expected[index]) * expected[index];
    maximum_absolute_error =
        std::max(maximum_absolute_error, std::abs(difference));
  }
  const double relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  std::printf("m=%d relative_l2=%.8g max_abs=%.8g\n", m, relative_l2,
              maximum_absolute_error);
  assert(relative_l2 < 2.0e-4);
  assert(maximum_absolute_error < 2.0e-3f);

  std::free(source_input_storage);
  std::free(decode_storage);
  std::free(amx_storage);
  std::free(reference_storage);
  std::free(weight_storage);
  std::free(input_storage);
}

}  // namespace

int main() {
  check_decoder();
  for (const int m : {1, 2, 4, 6, 8, 17, 32, 47, 64}) {
    check_shape(m);
  }
  assert(Kernel::amx_prefill_dispatches.load(std::memory_order_relaxed) == 9);
  assert(Kernel::avx512_prefill_dispatches.load(std::memory_order_relaxed) == 9);
  return 0;
}
