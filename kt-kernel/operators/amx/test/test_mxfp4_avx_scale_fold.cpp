#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../fp4-moe.hpp"

namespace {

using Kernel = amx::GemmKernel224MXFP4SmallKGroup;
using Mode = amx::MXFP4AVXScaleFoldMode;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void* allocate_aligned(size_t bytes) {
  const size_t rounded = (bytes + 63) & ~size_t{63};
  void* storage = std::aligned_alloc(64, rounded);
  if (storage == nullptr) throw std::bad_alloc();
  std::memset(storage, 0, rounded);
  return storage;
}

ggml_bf16_t to_bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
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

uint64_t fnv1a64(const std::vector<float>& values) {
  uint64_t hash = 14695981039346656037ULL;
  for (const float value : values) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint8_t>(bits >> (byte * 8));
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

void check_ocp_e8m0_semantics() {
  require(std::bit_cast<uint32_t>(Kernel::ue8m0_to_float(0)) == 0x00400000u,
          "OCP E8M0 byte 0 must decode to 2^-127");
  require(std::bit_cast<uint32_t>(Kernel::ue8m0_to_float(1)) == 0x00800000u,
          "OCP E8M0 byte 1 is wrong");
  require(std::bit_cast<uint32_t>(Kernel::ue8m0_to_float(254)) == 0x7f000000u,
          "OCP E8M0 byte 254 is wrong");
  require(std::isnan(Kernel::ue8m0_to_float(255)),
          "OCP E8M0 byte 255 must decode to NaN");
}

void check_all_safe_scale_and_code_pairs() {
  alignas(16) uint8_t packed_bytes[16];
  alignas(64) uint16_t lut_values[32];
  alignas(64) uint16_t exponent_values[32];
  for (int scale = Kernel::SCALE_FOLD_SAFE_MINIMUM;
       scale <= Kernel::SCALE_FOLD_SAFE_MAXIMUM; ++scale) {
    for (int packed = 0; packed < 256; ++packed) {
      std::fill(std::begin(packed_bytes), std::end(packed_bytes),
                static_cast<uint8_t>(packed));
      const __m128i input =
          _mm_load_si128(reinterpret_cast<const __m128i*>(packed_bytes));
      _mm512_store_si512(
          reinterpret_cast<__m512i*>(lut_values),
          Kernel::mxfp4_to_scaled_bf16_32_lut(
              input, static_cast<uint8_t>(scale)));
      _mm512_store_si512(
          reinterpret_cast<__m512i*>(exponent_values),
          Kernel::mxfp4_to_scaled_bf16_32_exponent(
              input, static_cast<uint8_t>(scale)));
      require(std::memcmp(lut_values, exponent_values, sizeof(lut_values)) == 0,
              "LUT and exponent scale folds differ");
      const int low_code = packed & 0x0f;
      const int high_code = packed >> 4;
      for (int lane = 0; lane < 32; ++lane) {
        const int code = lane % 2 == 0 ? low_code : high_code;
        const uint16_t base = Kernel::fp4_bf16_lut[code];
        const uint16_t expected =
            (base & 0x7fff) == 0
                ? base
                : static_cast<uint16_t>(
                      static_cast<int>(base) + (scale - 127) * 128);
        require(lut_values[lane] == expected,
                "scale-folded BF16 code is not exact");
      }
    }
  }
}

struct BufferStorage {
  void* storage;
  std::shared_ptr<Kernel::BufferB> buffer;

  BufferStorage(int n, int k, int group_size)
      : storage(allocate_aligned(
            Kernel::BufferB::required_size(n, k, group_size))),
        buffer(std::make_shared<Kernel::BufferB>(n, k, group_size, storage)) {}

  ~BufferStorage() {
    buffer.reset();
    std::free(storage);
  }
};

void check_whole_buffer_admission() {
  constexpr int n = 32;
  constexpr int k = 32;
  constexpr int group_size = 32;
  for (const Mode mode : {Mode::kLutV1, Mode::kExponentV1}) {
    BufferStorage safe(n, k, group_size);
    for (int row = 0; row < n; ++row) {
      safe.buffer->d[row] = static_cast<uint8_t>(
          Kernel::SCALE_FOLD_SAFE_MINIMUM +
          row % (Kernel::SCALE_FOLD_SAFE_MAXIMUM -
                 Kernel::SCALE_FOLD_SAFE_MINIMUM + 1));
    }
    safe.buffer->finalize_avx_scale_fold_domain(mode);
    require(safe.buffer->avx_scale_domain_finalized,
            "safe scale domain was not finalized");
    require(safe.buffer->avx_scale_fold_mode == mode,
            "safe whole-buffer scale domain was not admitted");
    require(safe.buffer->avx_scale_unsafe_count == 0,
            "safe scale domain reported unsafe bytes");
  }

  for (const uint8_t unsafe_scale : {uint8_t{0}, uint8_t{1}, uint8_t{253},
                                     uint8_t{254}, uint8_t{255}}) {
    BufferStorage unsafe(n, k, group_size);
    std::fill(unsafe.buffer->d, unsafe.buffer->d + n, uint8_t{120});
    unsafe.buffer->d[n - 1] = unsafe_scale;
    unsafe.buffer->finalize_avx_scale_fold_domain(Mode::kLutV1);
    require(unsafe.buffer->avx_scale_fold_mode == Mode::kOff,
            "unsafe whole-buffer scale domain did not fail closed");
    require(unsafe.buffer->avx_scale_unsafe_count == 1,
            "unsafe scale byte count is wrong");
    require(unsafe.buffer->avx_scale_nan_count ==
                static_cast<uint64_t>(unsafe_scale == 255),
            "E8M0 NaN count is wrong");
  }
}

std::vector<float> run_kernel(
    Mode mode, bool tiled, int m, int n, int k, int group_size,
    const std::vector<ggml_bf16_t>& source_input,
    const std::vector<uint8_t>& packed_weights,
    const std::vector<uint8_t>& scales) {
  void* input_storage =
      allocate_aligned(Kernel::BufferA::required_size(m, k));
  void* output_storage =
      allocate_aligned(Kernel::BufferC::required_size(m, n));
  BufferStorage weights(n, k, group_size);
  auto input = std::make_shared<Kernel::BufferA>(m, k, input_storage);
  auto output =
      std::make_shared<Kernel::BufferC>(m, n, output_storage);
  input->from_mat(m, const_cast<ggml_bf16_t*>(source_input.data()), 0, 1);
  weights.buffer->from_raw_mat(const_cast<uint8_t*>(packed_weights.data()), 0,
                               1);
  std::memcpy(weights.buffer->d, scales.data(), scales.size());
  weights.buffer->finalize_avx_scale_fold_domain(mode);
  if (tiled) {
    Kernel::fp4_mat_mat_kgroup(m, n, k, group_size, input.get(),
                               weights.buffer.get(), output.get(), 0, 1);
  } else {
    Kernel::fp4_mat_vec_kgroup(m, n, k, group_size, input.get(),
                               weights.buffer.get(), output.get(), 0, 1);
  }
  std::vector<float> result = logical_output(output.get(), m, n);
  std::free(output_storage);
  std::free(input_storage);
  return result;
}

void check_kernel_parity() {
  constexpr int n = 96;
  constexpr int k = 256;
  constexpr int group_size = 32;
  std::mt19937 generator(0x5343414cU);
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  std::uniform_int_distribution<int> scale_distribution(118, 126);
  std::uniform_real_distribution<float> activation_distribution(-0.25f,
                                                                 0.25f);
  std::vector<uint8_t> packed_weights(static_cast<size_t>(n) * k / 2);
  std::vector<uint8_t> scales(
      static_cast<size_t>(n) * (k / group_size));
  for (uint8_t& value : packed_weights) {
    value = static_cast<uint8_t>(byte_distribution(generator));
  }
  for (uint8_t& value : scales) {
    value = static_cast<uint8_t>(scale_distribution(generator));
  }

  for (int m = 1; m <= 6; ++m) {
    std::vector<ggml_bf16_t> source_input(static_cast<size_t>(m) * k);
    for (ggml_bf16_t& value : source_input) {
      value = to_bf16(activation_distribution(generator));
    }
    for (const bool tiled : {false, true}) {
      const std::vector<float> baseline = run_kernel(
          Mode::kOff, tiled, m, n, k, group_size, source_input,
          packed_weights, scales);
      const std::vector<float> lut = run_kernel(
          Mode::kLutV1, tiled, m, n, k, group_size, source_input,
          packed_weights, scales);
      const std::vector<float> exponent = run_kernel(
          Mode::kExponentV1, tiled, m, n, k, group_size, source_input,
          packed_weights, scales);
      require(lut == baseline,
              "lut-v1 output is not bit-identical to baseline");
      require(exponent == baseline,
              "exponent-v1 output is not bit-identical to baseline");
      std::printf(
          "mode_parity m=%d kernel=%s output_fnv1a64=%016llx\n", m,
          tiled ? "tiled" : "vec",
          static_cast<unsigned long long>(fnv1a64(baseline)));
    }
  }
}

}  // namespace

int main() {
  try {
    require(Kernel::N_BLOCK == KT_MXFP4_N_BLOCK,
            "compiled N_BLOCK telemetry differs from macro");
    check_ocp_e8m0_semantics();
    check_all_safe_scale_and_code_pairs();
    check_whole_buffer_admission();
    check_kernel_parity();
    std::printf("lut_identity=%.*s lut_fnv1a64=%s n_block=%d\n",
                static_cast<int>(Kernel::SCALE_FOLD_LUT_IDENTITY.size()),
                Kernel::SCALE_FOLD_LUT_IDENTITY.data(),
                Kernel::scale_fold_lut_hash().c_str(), Kernel::N_BLOCK);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "MXFP4 AVX scale-fold test failed: %s\n",
                 error.what());
    return 1;
  }
}
