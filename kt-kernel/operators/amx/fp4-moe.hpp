/**
 * @Description  : MXFP4 MoE operator — FP4 E2M1 weights × BF16 activations
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-04-20
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Based on k2-moe.hpp (RAWINT4). Key differences from RAWINT4:
 *   Weight:   FP4 E2M1 (nibble-packed, same layout) → PSHUFB lookup → BF16
 *   Act:      BF16 direct (BufferABF16Impl, no online INT8 quantization)
 *   Dot prod: _mm512_dpbf16_ps (BF16×BF16→FP32) instead of _mm512_dpbssd_epi32
 *   Scale:    FP32 per-group scale (weight only, no activation scale)
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_H

#include <atomic>
#include <cstdlib>

#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl
#include "moe_base.hpp"

namespace amx {

template <typename K>
struct BufferBMXFP4KGroupImpl {
  using dt = typename K::dt;
  dt* b;
  uint8_t* d;
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k, int k_group_size) {
    return sizeof(uint8_t) * n * k / 2 +
           sizeof(uint8_t) * n * (k / k_group_size);
  }

  BufferBMXFP4KGroupImpl(int n, int k, int k_group_size, void* ptr)
      : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % K_STEP == 0);
    if (n % N_STEP || k % K_STEP || k % k_group_size) {
      throw std::runtime_error(
          "BufferBMXFP4KGroupImpl dimensions are not kernel-aligned");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<uint8_t*>(offset_pointer(b, n * k / 2));
  }

  void from_raw_mat(uint8_t* proj, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    std::memcpy(reinterpret_cast<uint8_t*>(b) + n_start * row_bytes,
                proj + n_start * row_bytes,
                static_cast<size_t>(n_end - n_start) * row_bytes);
  }

  dt* get_submat(int, int k_arg, int n_begin, int k_begin) {
    const size_t row_bytes = static_cast<size_t>(k_arg) / 2;
    return reinterpret_cast<dt*>(
        reinterpret_cast<uint8_t*>(b) +
        static_cast<size_t>(n_begin) * row_bytes +
        static_cast<size_t>(k_begin) / 2);
  }

  uint8_t* get_scale(int, int n_begin, int k_arg, int k_begin) {
    return d + static_cast<size_t>(n_begin) * (k_arg / k_group_size) +
           k_begin / k_group_size;
  }
};

// ============================================================================
// MXFP4 kernel: FP4 E2M1 weights × BF16 activations → FP32 output (AVX512)
// ============================================================================
struct GemmKernel224MXFP4SmallKGroup {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "MXFP4_KGROUP"; }
  static int n_block_size(int) { return N_BLOCK; }
  static int recommended_nth(int n) {
    const int block_size = n_block_size(n);
    return (n + block_size - 1) / block_size;
  }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    const int block_size = n_block_size(n);
    int n_start = block_size * ith;
    int n_end = std::min(n, block_size * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {
#ifdef HAVE_AMX
    enable_amx();
    TileConfig tile_config;
    // TMM0-1: two 16x32 BF16 activation tiles.
    tile_config.set_row_col(0, 16, 32 * sizeof(ggml_bf16_t));
    tile_config.set_row_col(1, 16, 32 * sizeof(ggml_bf16_t));
    // TMM2-3: two VNNI-packed 32x16 BF16 weight tiles.
    tile_config.set_row_col(2, 16, 16 * 2 * sizeof(ggml_bf16_t));
    tile_config.set_row_col(3, 16, 16 * 2 * sizeof(ggml_bf16_t));
    // TMM4-7: the 2x2 grid of 16x16 FP32 accumulators.
    for (int tile = 4; tile < 8; ++tile) {
      tile_config.set_row_col(tile, 16, 16 * sizeof(float));
    }
    tile_config.set_config();
#endif
  }

  static inline std::atomic<uint64_t> avx512_decode_dispatches{0};
  static inline std::atomic<uint64_t> avx512_prefill_dispatches{0};
  static inline std::atomic<uint64_t> amx_prefill_dispatches{0};

  static int amx_minimum_expert_tokens() {
    static const int threshold = [] {
      constexpr int default_threshold = 8;
      const char* value = std::getenv("KT_MXFP4_AMX_MIN_EXPERT_TOKENS");
      if (value == nullptr || *value == '\0') return default_threshold;
      char* end = nullptr;
      const long parsed = std::strtol(value, &end, 10);
      if (end == value || *end != '\0' || parsed < 1 || parsed > 32) {
        return default_threshold;
      }
      return static_cast<int>(parsed);
    }();
    return threshold;
  }

  static int avx_tiled_minimum_expert_tokens() {
    static const int threshold = [] {
      constexpr int default_threshold = 4;
      const char* value =
          std::getenv("KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS");
      if (value == nullptr || *value == '\0') return default_threshold;
      char* end = nullptr;
      const long parsed = std::strtol(value, &end, 10);
      if (end == value || *end != '\0' || parsed < 1 || parsed > 32) {
        return default_threshold;
      }
      return static_cast<int>(parsed);
    }();
    return threshold;
  }

  // FP4 E2M1 → BF16 LUTs (16 entries each, for PSHUFB within 128-bit lanes)
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,   //  0..7  positive
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};  //  8..15 negative
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
      0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,   //  0..7  positive
      0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};  //  8..15 negative
  alignas(64) static constexpr uint16_t fp4_bf16_lut[32] = {
      0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
      0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0,
      0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
      0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0};

  // Original byte-shuffle decoder retained as an independent test oracle.
  __attribute__((always_inline)) static inline __m512i
  mxfp4_to_bf16_32_reference(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    __m128i lut_lo = _mm_load_si128((__m128i*)fp4_bf16_lo);
    __m128i lut_hi = _mm_load_si128((__m128i*)fp4_bf16_hi);

    // Look up low/high bytes for lo nibbles → 16 BF16 values
    __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
    __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
    __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);  // BF16(lo[0..7])
    __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);  // BF16(lo[8..15])

    // Look up low/high bytes for hi nibbles → 16 BF16 values
    __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
    __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
    __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);  // BF16(hi[0..7])
    __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);  // BF16(hi[8..15])

    // Interleave lo/hi at 16-bit: [lo[0],hi[0], lo[1],hi[1], ...] = column order
    __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);  // cols  0..7
    __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);  // cols  8..15
    __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);  // cols 16..23
    __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);  // cols 24..31

    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
  }

  // Convert 16 packed FP4 bytes (32 values = 1 K-group) to 32 BF16 values.
  // VPERMW maps the expanded nibble indices directly to complete BF16 values,
  // replacing four byte-table shuffles plus four unpack stages in the decode
  // hot loop. Output order remains lo[0], hi[0], ..., lo[15], hi[15].
  __attribute__((always_inline)) static inline __m512i
  mxfp4_to_bf16_32(__m128i packed) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(packed, nibble_mask);
    const __m128i high =
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
    const __m128i indices_low = _mm_unpacklo_epi8(low, high);
    const __m128i indices_high = _mm_unpackhi_epi8(low, high);
    const __m256i byte_indices = _mm256_inserti128_si256(
        _mm256_castsi128_si256(indices_low), indices_high, 1);
    const __m512i word_indices = _mm512_cvtepu8_epi16(byte_indices);
    const __m512i lookup =
        _mm512_load_si512(reinterpret_cast<const __m512i*>(fp4_bf16_lut));
    return _mm512_permutexvar_epi16(word_indices, lookup);
  }

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
    inline static const __m512i odd_mask = _mm512_set1_epi32(0xFFFF0000);
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, odd_mask));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d;
#else
    __m512 w_even;
    __m512 w_odd;
    inline static const __m128i lo_mask = _mm_set1_epi8(0x0F);
    inline static const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f,
                                                    -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
#endif

    __attribute__((always_inline)) DequantizedWeight(__m128i w) {
#if defined(__AVX512BF16__)
      d = (__m512bh)mxfp4_to_bf16_32(w);
#else
      __m128i lo = _mm_and_si128(w, lo_mask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), lo_mask);

      __m512i lo_32 = _mm512_cvtepu8_epi32(lo);
      __m512i hi_32 = _mm512_cvtepu8_epi32(hi);

      w_even = _mm512_permutexvar_ps(lo_32, lut);
      w_odd = _mm512_permutexvar_ps(hi_32, lut);
#endif
    }
  };

  __attribute__((always_inline)) static inline __m512 mxfp4_dot_bf16(const DequantizedWeight& w,
                                                                     const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(_mm512_setzero_ps(), act.a, w.d);
#else
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
#endif
  }

  // Buffers
  using BufferA = BufferABF16Impl<GemmKernel224MXFP4SmallKGroup>;        // raw BF16, no quant
  using BufferB = BufferBMXFP4KGroupImpl<GemmKernel224MXFP4SmallKGroup>;
  using BufferC = BufferCReduceImpl<GemmKernel224MXFP4SmallKGroup>;      // FP32 reduce

  // 4 个 zmm 的 horizontal reduce → 4 个连续 fp32。
  // 4 次 reduce_add_ps 之间无依赖，编译器/CPU 可并行调度。
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // Eight independent reductions let the out-of-order core overlap the
  // shuffle/add trees while amortizing one activation load across twice as
  // many native-MXFP4 output rows.
  __attribute__((always_inline)) static inline void reduce8(
      __m512 s0, __m512 s1, __m512 s2, __m512 s3, __m512 s4, __m512 s5,
      __m512 s6, __m512 s7, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
    dst[4] = _mm512_reduce_add_ps(s4);
    dst[5] = _mm512_reduce_add_ps(s5);
    dst[6] = _mm512_reduce_add_ps(s6);
    dst[7] = _mm512_reduce_add_ps(s7);
  }

  __attribute__((always_inline)) static inline float ue8m0_to_float(
      uint8_t scale) {
    uint32_t bits = static_cast<uint32_t>(scale) << 23;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  // mat-vec: M 个独立 token，N 维 4 行一组累加，摊销 horizontal reduce。
  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    const uint64_t previous_dispatches =
        avx512_decode_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (previous_dispatches == 0) {
      std::fprintf(stderr,
                   "[KT][MXFP4] first AVX-512 decode dispatch "
                   "(m=%d n=%d k=%d group=%d thread=%d/%d)\n",
                   m, n, k, k_group_size, ith, nth);
    }
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      // Main loop: eight output rows share one activation decode. Keeping
      // each row's K-group order unchanged makes this bit-identical to the
      // four-row kernel while reducing activation and loop overhead.
      for (; n_pos + 8 <= n_end; n_pos += 8) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        __m128i* w4 = (__m128i*)bb->get_submat(n, k, n_pos + 4, 0);
        __m128i* w5 = (__m128i*)bb->get_submat(n, k, n_pos + 5, 0);
        __m128i* w6 = (__m128i*)bb->get_submat(n, k, n_pos + 6, 0);
        __m128i* w7 = (__m128i*)bb->get_submat(n, k, n_pos + 7, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        const uint8_t* s4 = bb->get_scale(n, n_pos + 4, k, 0);
        const uint8_t* s5 = bb->get_scale(n, n_pos + 5, k, 0);
        const uint8_t* s6 = bb->get_scale(n, n_pos + 6, k, 0);
        const uint8_t* s7 = bb->get_scale(n, n_pos + 7, k, 0);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const DequantizedWeight d4(w4[g]);
          const DequantizedWeight d5(w5[g]);
          const DequantizedWeight d6(w6[g]);
          const DequantizedWeight d7(w7[g]);
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s0[g])), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s1[g])), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s2[g])), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s3[g])), mxfp4_dot_bf16(d3, a), acc3);
          acc4 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s4[g])), mxfp4_dot_bf16(d4, a), acc4);
          acc5 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s5[g])), mxfp4_dot_bf16(d5, a), acc5);
          acc6 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s6[g])), mxfp4_dot_bf16(d6, a), acc6);
          acc7 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s7[g])), mxfp4_dot_bf16(d7, a), acc7);
        }
        reduce8(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                c_row + (n_pos - n_start));
      }
      // Preserve support for thread partitions that are only four-row
      // aligned. Current DSV4 partitions are 256 rows and use only the loop
      // above.
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s0[g])), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s1[g])), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s2[g])), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s3[g])), mxfp4_dot_bf16(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      // N 尾巴: N % 4 != 0 时单行 fallback
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s[g])), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

  // mat-mat: 4×4 register tile (M_TILE=4, N_TILE=4 → 16 累加器)。
  // 每 K-group 解码 4 行 N 一次, 被 4 个 token 共享 → PSHUFB 解码开销 / 4。
  // M / N 尾巴回退到 mat-vec 单 token 内层 (V4 chunked-prefill 16/32/64 整数倍, 极少触发)。
  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    const uint64_t previous_dispatches =
        avx512_prefill_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (previous_dispatches == 0) {
      std::fprintf(stderr,
                   "[KT][MXFP4] first AVX-512 prefill dispatch "
                   "(m=%d n=%d k=%d group=%d thread=%d/%d threshold=%d)\n",
                   m, n, k, k_group_size, ith, nth,
                   avx_tiled_minimum_expert_tokens());
    }
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      __m512bh* a_rows[MB] = {
          (__m512bh*)ba->get_submat(m, k, m_pos + 0, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 1, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 2, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          // 4 行权重解码一次, MB 个 token 共享
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = _mm512_set1_ps(ue8m0_to_float(s0[g]));
          const __m512 sv1 = _mm512_set1_ps(ue8m0_to_float(s1[g]));
          const __m512 sv2 = _mm512_set1_ps(ue8m0_to_float(s2[g]));
          const __m512 sv3 = _mm512_set1_ps(ue8m0_to_float(s3[g]));

#define V_FMA_ROW(M_I)                                                      \
  do {                                                                      \
    const ActivationBF16 a(a_rows[M_I][g]);                                 \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp4_dot_bf16(d0, a), acc[M_I][0]); \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp4_dot_bf16(d1, a), acc[M_I][1]); \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp4_dot_bf16(d2, a), acc[M_I][2]); \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp4_dot_bf16(d3, a), acc[M_I][3]); \
  } while (0)
          V_FMA_ROW(0);
          V_FMA_ROW(1);
          V_FMA_ROW(2);
          V_FMA_ROW(3);
#undef V_FMA_ROW
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      // N 尾巴: 单 N 列 × MB token (V4 不触发)
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(ue8m0_to_float(s[g])), mxfp4_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
        }
      }
    }
    // Share every decoded weight row across the remaining 1--3 tokens.
    // DSpark frequently sends two or three correlated rows to an expert; the
    // prior fallback decoded and streamed the same native-MXFP4 row once per
    // token. Each output keeps the identical K-group/FMA order.
    const int remaining_m = m - m_pos;
    if (remaining_m > 0) {
      __m512bh* a_rows[MB] = {};
      for (int i = 0; i < remaining_m; ++i) {
        a_rows[i] =
            (__m512bh*)ba->get_submat(m, k, m_pos + i, 0);
      }
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 acc[MB][NB];
        for (int i = 0; i < remaining_m; ++i) {
          for (int j = 0; j < NB; ++j) {
            acc[i][j] = _mm512_setzero_ps();
          }
        }
        for (int g = 0; g < kg_count; g++) {
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 =
              _mm512_set1_ps(ue8m0_to_float(s0[g]));
          const __m512 sv1 =
              _mm512_set1_ps(ue8m0_to_float(s1[g]));
          const __m512 sv2 =
              _mm512_set1_ps(ue8m0_to_float(s2[g]));
          const __m512 sv3 =
              _mm512_set1_ps(ue8m0_to_float(s3[g]));
          for (int i = 0; i < remaining_m; ++i) {
            const ActivationBF16 a(a_rows[i][g]);
            acc[i][0] = _mm512_fmadd_ps(
                sv0, mxfp4_dot_bf16(d0, a), acc[i][0]);
            acc[i][1] = _mm512_fmadd_ps(
                sv1, mxfp4_dot_bf16(d1, a), acc[i][1]);
            acc[i][2] = _mm512_fmadd_ps(
                sv2, mxfp4_dot_bf16(d2, a), acc[i][2]);
            acc[i][3] = _mm512_fmadd_ps(
                sv3, mxfp4_dot_bf16(d3, a), acc[i][3]);
          }
        }
        for (int i = 0; i < remaining_m; ++i) {
          float* c_row =
              bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(
              acc[i][0], acc[i][1], acc[i][2], acc[i][3],
              c_row + (n_pos - n_start));
        }
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc[MB];
        for (int i = 0; i < remaining_m; ++i) {
          acc[i] = _mm512_setzero_ps();
        }
        for (int g = 0; g < kg_count; g++) {
          const DequantizedWeight d(w[g]);
          const __m512 sv =
              _mm512_set1_ps(ue8m0_to_float(s[g]));
          for (int i = 0; i < remaining_m; ++i) {
            const ActivationBF16 a(a_rows[i][g]);
            acc[i] = _mm512_fmadd_ps(
                sv, mxfp4_dot_bf16(d, a), acc[i]);
          }
        }
        for (int i = 0; i < remaining_m; ++i) {
          float* c_row =
              bc->get_submat(m, n, m_pos + i, n_start);
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc[i]);
        }
      }
    }
  }

  /**
   * High-M native-MXFP4 kernel.
   *
   * The persisted BufferB remains nibble-packed E2M1 plus the checkpoint's
   * native one-byte UE8M0 scale. For each K=32 group we transiently
   * decode and scale a 32x32 weight block into two VNNI BF16 tiles.  The
   * decoded block is then shared by 32 activation rows and four AMX BF16 dot
   * products.  Accumulators remain resident across the entire K dimension,
   * avoiding the scale/dot/store cycle that made the earlier per-group AMX
   * proposal slower than the direct AVX-512 path.
   */
  __attribute__((always_inline)) static inline __m512i scale_bf16_32(__m512i values, float scale) {
    // Native MXFP4 scales are UE8M0 powers of two. For normal finite results,
    // multiplying an E2M1 codepoint by that scale changes only the BF16
    // exponent, so perform the exact exponent adjustment directly. This
    // removes two BF16->FP32 expansions, two FP32 multiplies, and one packed
    // FP32->BF16 conversion from every decoded 32x32 AMX weight block.
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    const uint8_t scale_exponent = static_cast<uint8_t>(scale_bits >> 23);
    if (scale_exponent >= 2 && scale_exponent <= 252) {
      const __m512i magnitude_mask = _mm512_set1_epi16(0x7fff);
      const __m512i magnitudes = _mm512_and_si512(values, magnitude_mask);
      const __mmask32 nonzero = _mm512_cmpneq_epi16_mask(
          magnitudes, _mm512_setzero_si512());
      const int16_t exponent_delta = static_cast<int16_t>(
          (static_cast<int>(scale_exponent) - 127) * 128);
      return _mm512_mask_add_epi16(
          values, nonzero, values, _mm512_set1_epi16(exponent_delta));
    }

    // Preserve IEEE handling for exotic scales that could create BF16
    // subnormals or infinities. DeepSeek V4's native scales are 119..124 and
    // always take the exponent-only path above.
    const __m256i low_bf16 = _mm512_castsi512_si256(values);
    const __m256i high_bf16 = _mm512_extracti64x4_epi64(values, 1);
    const __m512 low_fp32 =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(low_bf16), 16));
    const __m512 high_fp32 =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(high_bf16), 16));
    const __m512 scale_vector = _mm512_set1_ps(scale);
    return (__m512i)_mm512_cvtne2ps_pbh(_mm512_mul_ps(high_fp32, scale_vector),
                                        _mm512_mul_ps(low_fp32, scale_vector));
  }

  static void fp4_mat_mat_amx_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc,
                                     int ith, int nth) {
#ifndef HAVE_AMX
    fp4_mat_mat_kgroup(m, n, k, k_group_size, ba, bb, bc, ith, nth);
#else
    if (m == 0) return;
    if (k_group_size != 32 || k % 32 != 0) {
      throw std::runtime_error("MXFP4 AMX prefill requires K-group size 32 and K aligned to 32");
    }

    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    if (n_start % 32 != 0 || n_end % 32 != 0) {
      throw std::runtime_error("MXFP4 AMX prefill requires N ranges aligned to 32");
    }
    const uint64_t previous_dispatches =
        amx_prefill_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (previous_dispatches == 0) {
      std::fprintf(stderr,
                   "[KT][MXFP4] first AMX prefill dispatch "
                   "(m=%d n=%d k=%d group=%d thread=%d/%d threshold=%d)\n",
                   m, n, k, k_group_size, ith, nth,
                   amx_minimum_expert_tokens());
    }

    constexpr int M_TILE = 32;
    constexpr int N_TILE = 32;
    constexpr int K_TILE = 32;
    constexpr int HALF_TILE = 16;
    alignas(64) ggml_bf16_t activation_tiles[M_TILE * K_TILE];
    alignas(64) ggml_bf16_t weight_tiles[N_TILE * K_TILE];
    alignas(64) float tail_output[M_TILE * N_TILE];

    for (int m_pos = 0; m_pos < m; m_pos += M_TILE) {
      const int valid_m = std::min(M_TILE, m - m_pos);
      // DSpark verification routes at most six rows to one expert.  The
      // original tail path padded those rows to 32 and issued both halves of
      // the 32x32 tile grid, even though rows 16..31 could never contribute
      // to a live output.  Keep the same K-group and FP32 accumulation order
      // for real rows, but omit the second activation tile and its two dot
      // products when the tail fits in one 16-row AMX tile.
      const bool single_m_tile = valid_m <= HALF_TILE;
      for (int n_pos = n_start; n_pos < n_end; n_pos += N_TILE) {
        _tile_zero(4);
        _tile_zero(5);
        if (!single_m_tile) {
          _tile_zero(6);
          _tile_zero(7);
        }

        for (int k_pos = 0; k_pos < k; k_pos += K_TILE) {
          if (valid_m != M_TILE) {
            const size_t padded_rows =
                single_m_tile ? HALF_TILE : M_TILE;
            std::memset(
                activation_tiles, 0,
                padded_rows * K_TILE * sizeof(ggml_bf16_t));
          }
          for (int row = 0; row < valid_m; ++row) {
            const auto* source = reinterpret_cast<const __m512i*>(
                ba->get_submat(m, k, m_pos + row, k_pos));
            _mm512_store_si512(
                reinterpret_cast<__m512i*>(activation_tiles + row * K_TILE),
                _mm512_load_si512(source));
          }

          for (int row = 0; row < N_TILE; ++row) {
            const auto* packed = reinterpret_cast<const __m128i*>(
                bb->get_submat(n, k, n_pos + row, k_pos));
            const float scale = ue8m0_to_float(
                *bb->get_scale(n, n_pos + row, k, k_pos));
            const __m512i decoded = mxfp4_to_bf16_32(_mm_loadu_si128(packed));
            _mm512_store_si512(
                reinterpret_cast<__m512i*>(weight_tiles + row * K_TILE),
                scale_bf16_32(decoded, scale));
          }
          // Each transpose turns 16 logical N rows of 32 BF16 K values into
          // the 16x64-byte VNNI form consumed by TILEDPBF16PS.
          transpose_16x16_32bit(reinterpret_cast<__m512i*>(weight_tiles));
          transpose_16x16_32bit(
              reinterpret_cast<__m512i*>(weight_tiles + HALF_TILE * K_TILE));

          _tile_loadd(0, activation_tiles, K_TILE * sizeof(ggml_bf16_t));
          _tile_loadd(2, weight_tiles, HALF_TILE * 2 * sizeof(ggml_bf16_t));
          _tile_loadd(3, weight_tiles + HALF_TILE * K_TILE,
                      HALF_TILE * 2 * sizeof(ggml_bf16_t));
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          if (!single_m_tile) {
            _tile_loadd(1, activation_tiles + HALF_TILE * K_TILE,
                        K_TILE * sizeof(ggml_bf16_t));
            _tile_dpbf16ps(6, 1, 2);
            _tile_dpbf16ps(7, 1, 3);
          }
        }

        const int output_block_size = n_block_size(n);
        const int n_block_begin =
            n_pos / output_block_size * output_block_size;
        const int n_block_size =
            std::min(output_block_size, n - n_block_begin);
        if (valid_m == M_TILE) {
          float* output = bc->get_submat(m, n, m_pos, n_pos);
          const size_t output_stride = static_cast<size_t>(n_block_size) * sizeof(float);
          _tile_stored(4, output, output_stride);
          _tile_stored(5, output + HALF_TILE, output_stride);
          _tile_stored(6, output + HALF_TILE * n_block_size, output_stride);
          _tile_stored(7, output + HALF_TILE * n_block_size + HALF_TILE, output_stride);
        } else {
          constexpr size_t tail_stride = N_TILE * sizeof(float);
          _tile_stored(4, tail_output, tail_stride);
          _tile_stored(5, tail_output + HALF_TILE, tail_stride);
          if (!single_m_tile) {
            _tile_stored(6, tail_output + HALF_TILE * N_TILE, tail_stride);
            _tile_stored(
                7, tail_output + HALF_TILE * N_TILE + HALF_TILE,
                tail_stride);
          }
          for (int row = 0; row < valid_m; ++row) {
            float* output_row = bc->get_submat(m, n, m_pos + row, n_pos);
            std::memcpy(output_row, tail_output + row * N_TILE,
                        N_TILE * sizeof(float));
          }
        }
      }
    }
#endif
  }
};

// Dispatch functions
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  using Kernel = GemmKernel224MXFP4SmallKGroup;
#ifdef HAVE_AMX
  if (m >= Kernel::amx_minimum_expert_tokens()) {
    Kernel::fp4_mat_mat_amx_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
  Kernel::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// AMX_FP4_MOE_TP — CRTP class, identical structure to AMX_K2_MOE_TP
// ============================================================================
template <class T = amx::GemmKernel224MXFP4SmallKGroup>
class AMX_FP4_MOE_TP : public AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP4_MOE_TP() = default;
  AMX_FP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("MXFP4 MoE only supports KGroup FP4");
    }
    printf("Creating AMX_FP4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_FP4_MOE_TP() = default;

  // BufferA: raw BF16, no group_size needed
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok ||
        m >= T::avx_tiled_minimum_expert_tokens()) {
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok ||
        m >= T::avx_tiled_minimum_expert_tokens()) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("MXFP4 MoE only support KGroup FP4.");
    if (config_.gate_scale == nullptr) throw std::runtime_error("MXFP4 MoE only support load native weight.");

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;
          std::memcpy(gate_bb_[expert_idx]->d,
                      (uint8_t*)config_.gate_scale +
                          (logical_expert_id * scale_elem_count),
                      scale_elem_count);
          std::memcpy(up_bb_[expert_idx]->d,
                      (uint8_t*)config_.up_scale +
                          (logical_expert_id * scale_elem_count),
                      scale_elem_count);
          std::memcpy(down_bb_[expert_idx]->d,
                      (uint8_t*)config_.down_scale +
                          (logical_expert_id * scale_elem_count),
                      scale_elem_count);
        },
        nullptr);
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      __m512i data = _mm512_loadu_si512((__m512i*)s);
      _mm512_storeu_si512((__m512i*)d, data);
      d += 64;
      s += 64;
    }
    if (bytes -= chunks * 64) std::memcpy(d, s, bytes);
  }

  static inline void fast_fp32_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
      __m512 v0 = _mm512_loadu_ps(src + i);
      __m512 v1 = _mm512_loadu_ps(src + i + 16);
      __m512i i0 = _mm512_srli_epi32(_mm512_castps_si512(v0), 16);
      __m512i i1 = _mm512_srli_epi32(_mm512_castps_si512(v1), 16);
      __m512i packed = _mm512_packus_epi32(i0, i1);
      __m512i permuted = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);
      _mm512_storeu_si512((__m512i*)(dst + i), permuted);
    }
    for (; i < count; i++) dst[i] = ggml_fp32_to_bf16(src[i]);
  }

  static inline void fast_ue8m0_to_bf16(
      ggml_bf16_t* __restrict dst, const uint8_t* __restrict src,
      size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
      const __m256i exponent_bytes =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      const __m512i bf16_bits =
          _mm512_slli_epi16(_mm512_cvtepu8_epi16(exponent_bytes), 7);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), bf16_bits);
    }
    for (; i < count; ++i) {
      dst[i] = static_cast<ggml_bf16_t>(
          static_cast<uint16_t>(src[i]) << 7);
    }
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;

      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size, offset_in_gpu_weight, offset_in_gpu_scale,
           gpu_tp_weight_bytes, gpu_tp_scale_elem_count, w13_weight_dst, w13_scale_dst, w2_weight_dst, w2_scale_dst,
           group_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_id;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + start, (uint8_t*)gate_bb_[expert_id]->b + start,
                            end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; col++) {
                fast_memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            (uint8_t*)down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);
                fast_ue8m0_to_bf16(
                    w2_scale_dst + col * gpu_scale_stride +
                        gpu_scale_slice_offset,
                    down_bb_[expert_id]->d + col * scale_per_col,
                    scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_ue8m0_to_bf16(w13_scale_dst + offset_in_gpu_scale,
                                  gate_bb_[expert_id]->d,
                                  cpu_tp_scale_elem_count);
            } else {
              fast_ue8m0_to_bf16(
                  w13_scale_dst + offset_in_gpu_scale +
                      gpu_tp_scale_elem_count,
                  up_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;

      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size, gpu_tp_weight_bytes, gpu_tp_scale_elem_count,
           group_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_type;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + start, (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight + start,
                            end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

                fast_memcpy(w2_weight_dst + col * weight_per_gpu_col,
                            (uint8_t*)down_bb_[expert_id]->b + col_offset_weight, weight_per_gpu_col);
                fast_ue8m0_to_bf16(
                    w2_scale_dst + col * scale_per_gpu_col,
                    down_bb_[expert_id]->d + col_offset_scale,
                    scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_ue8m0_to_bf16(
                  w13_scale_dst,
                  gate_bb_[expert_id]->d + cpu_offset_scale,
                  data_per_gpu_tp_scale);
            } else {
              fast_ue8m0_to_bf16(
                  w13_scale_dst + gpu_tp_scale_elem_count,
                  up_bb_[expert_id]->d + cpu_offset_scale,
                  data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_FP4_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AMX_FP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 MoE only supports Packed FP4 with KGroup Scale");

    printf("From %s\n", use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "Packed FP4 with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new uint8_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new uint8_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new uint8_t[tpc.expert_num * scales_elem_count];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              uint8_t* src_gate_scale =
                  (uint8_t*)config.gate_scales[0][expert_id];
              uint8_t* src_up_scale =
                  (uint8_t*)config.up_scales[0][expert_id];
              uint8_t* src_down_scale =
                  (uint8_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), scales_elem_count);
              memcpy((uint8_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((uint8_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       tpc.intermediate_size / group_size);
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (uint8_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       scales_elem_count);
                memcpy((uint8_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (uint8_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (tpc.intermediate_size >> 1));
                  memcpy((uint8_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (uint8_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         tpc.intermediate_size / group_size);
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done.\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (uint8_t*)(tpc.gate_scale);
      delete[] (uint8_t*)(tpc.up_scale);
      delete[] (uint8_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP4_MOE_H
