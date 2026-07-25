/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_H
#define CPUINFER_OPERATOR_AMX_MOE_H

// #define CHECK
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#include <algorithm>
#include <cstring>
#include <vector>

#include "moe_base.hpp"

template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 protected:
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
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

#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
#endif

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                            size_t scale_size) {
    // printf("expert %d, size %ld, scale size %ld\n", expert_idx, size, scale_size);
    // std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file: %s", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                            std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                     .c_str());
      // throw std::runtime_error("No such file");
    }
    of.write((char*)bb, size - scale_size);
    of.close();
    // of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file\n");
      // throw std::runtime_error("No such file");
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    // std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * (size - scale_size) / mat_split);
    f.read(((char*)bb) + mat_split_idex * (size - scale_size) / mat_split, (size - scale_size) / mat_split);
    f.close();
    // f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                     "_scale_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(scale_size) + "Byte" + "_scale_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * scale_size / mat_split);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_size / mat_split, scale_size / mat_split);
  }

  inline void bind_external_readonly_weight(const std::shared_ptr<typename T::BufferB>& buffer, const void* weight,
                                            const void* scale) {
    if constexpr (requires(typename T::BufferB& candidate) { candidate.set_external_readonly_data(weight, scale); }) {
      buffer->set_external_readonly_data(weight, scale);
    } else {
      throw std::runtime_error("selected AMX BufferB does not support shared read-only host weights");
    }
  }
#ifdef CHECK
  inline void load_check() {
    memcpy(check_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
  }

  void verify_load_right() {
    // printf("varify down bb_0 %d\n", tp_part_idx);
    memcpy(verify_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
    // check if verify_bb_0 equal to check_bb_0
    if (memcmp(verify_bb, check_bb, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)) != 0) {
      printf("verify error\n");
      for (size_t i = 0; i < T::BufferB::required_size(config_.hidden_size, config_.intermediate_size); ++i) {
        if (verify_bb[i] != check_bb[i]) {
          printf("Difference at byte %zu: verify_bb_%d[%zu] = %02x, check_bb[%zu] = %02x\n", i, compare_expers, i,
                 (unsigned char)verify_bb[i], i, (unsigned char)check_bb[i]);
          break;  // find the first difference and exit
        }
      }
      assert(0);
    } else {
      printf("pass verify\n");
      // pick out the 100th~150th byte of scale to see
      printf("numa %d, verify_bb_%d:\n", tp_part_idx, compare_expers);
      size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      size_t scale_size = config_.hidden_size * sizeof(float);
      for (size_t i = size - scale_size; i < size - scale_size + 50; ++i) {
        printf("%02x ", (unsigned char)verify_bb[i]);
      }
      printf("\n");
    }
  }
#endif

#ifdef FORWARD_TIME_REPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> last_now;
#endif

 public:
  AMX_MOE_TP() = default;

  AMX_MOE_TP(GeneralMOEConfig config, int tp_part_idx = 0) : Base(config, tp_part_idx) {
    // Initialization now happens in derived_init() which is called by base constructor
  }

  void derived_init() {
    printf("Creating AMX_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
    auto& load = config_.load;
    auto& save = config_.save;

    std::filesystem::path prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
    if (save) {
      std::cout << "Creating " << prefix << std::endl;
      std::filesystem::create_directories(prefix);
    }
    if (load) {
      if (std::filesystem::exists(prefix)) {
        std::cout << "Loading from " << prefix << std::endl;
      } else {
        throw std::runtime_error("Path not found: " + prefix.string());
      }
    }
  }

  ~AMX_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - no group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = down_ba_[expert_idx];
    auto& bb = down_bb_[expert_idx];
    auto& bc = down_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }
  }

  /**
   * @brief Dequantize one resident AMXINT4 expert into a BF16 GPU staging
   * buffer.
   *
   * The ordinary AMXINT4 decode path keeps its tile-packed INT4 weights in
   * host DRAM.  Long-prompt stream-loading prefill uses the same
   * authoritative allocation, but temporarily materializes a small expert
   * chunk as BF16 on each tensor-parallel GPU.  BufferBInt4Impl::to_mat()
   * reverses the AMX/VNNI packing and applies the stored per-output scale.
   *
   * The ordinary path preserves the one-to-one CPU/GPU TP mapping.  SmallEP
   * additionally admits gpu_tp_count=1: every CPU/NUMA part dequantizes its
   * shard, and the shards are assembled into one complete BF16 expert owned
   * by a single GPU.
   */
  void write_weights_to_bf16_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id,
                                    const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs) const {
    const bool assemble_complete_expert = gpu_tp_count == 1 && cpu_tp_count > 1;
    if (gpu_tp_count != cpu_tp_count && !assemble_complete_expert) {
      throw std::runtime_error("AMXINT4 BF16 export requires GPU TP to match CPU TP or equal 1");
    }
    if (tp_part_idx < 0 || tp_part_idx >= cpu_tp_count) {
      throw std::runtime_error("AMXINT4 BF16 export TP part is out of range");
    }
    if (expert_id < 0 || expert_id >= config_.expert_num) {
      throw std::runtime_error("AMXINT4 BF16 export expert_id is out of range");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count ||
        w13_weight_ptrs[assemble_complete_expert ? 0 : tp_part_idx] == 0 ||
        w2_weight_ptrs[assemble_complete_expert ? 0 : tp_part_idx] == 0) {
      throw std::runtime_error("AMXINT4 BF16 export requires one non-null weight pointer per GPU TP rank");
    }

    const int cpu_intermediate_size = full_config.intermediate_size / cpu_tp_count;
    if (full_config.intermediate_size % cpu_tp_count != 0 || cpu_intermediate_size != config_.intermediate_size ||
        full_config.hidden_size != config_.hidden_size) {
      throw std::runtime_error("AMXINT4 BF16 export CPU/GPU tensor partitions do not match");
    }

    const int destination_index = assemble_complete_expert ? 0 : tp_part_idx;
    auto* w13 = reinterpret_cast<ggml_bf16_t*>(w13_weight_ptrs[destination_index]);
    auto* w2 = reinterpret_cast<ggml_bf16_t*>(w2_weight_ptrs[destination_index]);
    const size_t projection_elements = (size_t)config_.intermediate_size * config_.hidden_size;
    const size_t full_projection_elements = (size_t)full_config.intermediate_size * config_.hidden_size;
    auto* gate_destination = w13 + (assemble_complete_expert ? (size_t)tp_part_idx * projection_elements : 0);
    auto* up_destination =
        w13 + (assemble_complete_expert ? full_projection_elements + (size_t)tp_part_idx * projection_elements
                                        : projection_elements);
    std::vector<ggml_bf16_t> local_down;
    auto* down_destination = w2;
    if (assemble_complete_expert) {
      local_down.resize(projection_elements);
      down_destination = local_down.data();
    }

    const int gate_up_tasks = T::recommended_nth(config_.intermediate_size);
    const int down_tasks = T::recommended_nth(config_.hidden_size);
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if constexpr (requires(typename T::BufferB& buffer, ggml_bf16_t* destination) {
                    buffer.to_mat(destination, 0, 1);
                  }) {
      pool->do_work_stealing_job(
          gate_up_tasks * 2 + down_tasks, nullptr,
          [=, this](int task_id) {
            if (task_id < gate_up_tasks) {
              gate_bb_[expert_id]->to_mat(gate_destination, task_id, gate_up_tasks);
            } else if (task_id < gate_up_tasks * 2) {
              const int ith = task_id - gate_up_tasks;
              up_bb_[expert_id]->to_mat(up_destination, ith, gate_up_tasks);
            } else {
              const int ith = task_id - gate_up_tasks * 2;
              down_bb_[expert_id]->to_mat(down_destination, ith, down_tasks);
            }
          },
          nullptr);
      if (assemble_complete_expert) {
        pool->do_work_stealing_job(
            down_tasks, nullptr,
            [this, w2, &local_down, &full_config, down_tasks](int task_id) {
              for (int row = task_id; row < config_.hidden_size; row += down_tasks) {
                std::memcpy(
                    w2 + (size_t)row * full_config.intermediate_size + (size_t)tp_part_idx * config_.intermediate_size,
                    local_down.data() + (size_t)row * config_.intermediate_size,
                    (size_t)config_.intermediate_size * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      }
    } else {
      throw std::runtime_error("selected AMXINT4 BufferB cannot export BF16 stream weights");
    }
  }

  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (config_.share_host_weights && config_.gate_projs.empty()) {
      throw std::runtime_error("shared AMXINT4 weights require an external pointer table");
    }
    if (config_.gate_projs.size()) {
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_id) {
            // printf("Load layer %d [%d/%d]\n", config_.layer_idx, expert_id, config_.expert_num);
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
            if (config_.share_host_weights) {
              if (config_.gate_scales.size() <= static_cast<size_t>(tp_part_idx) ||
                  config_.up_scales.size() <= static_cast<size_t>(tp_part_idx) ||
                  config_.down_scales.size() <= static_cast<size_t>(tp_part_idx) ||
                  config_.gate_projs[tp_part_idx].size() <= logical_expert_id ||
                  config_.up_projs[tp_part_idx].size() <= logical_expert_id ||
                  config_.down_projs[tp_part_idx].size() <= logical_expert_id ||
                  config_.gate_scales[tp_part_idx].size() <= logical_expert_id ||
                  config_.up_scales[tp_part_idx].size() <= logical_expert_id ||
                  config_.down_scales[tp_part_idx].size() <= logical_expert_id) {
                throw std::runtime_error("incomplete shared AMXINT4 expert pointer table");
              }
              bind_external_readonly_weight(gate_bb_[expert_id], config_.gate_projs[tp_part_idx][logical_expert_id],
                                            config_.gate_scales[tp_part_idx][logical_expert_id]);
              bind_external_readonly_weight(up_bb_[expert_id], config_.up_projs[tp_part_idx][logical_expert_id],
                                            config_.up_scales[tp_part_idx][logical_expert_id]);
              bind_external_readonly_weight(down_bb_[expert_id], config_.down_projs[tp_part_idx][logical_expert_id],
                                            config_.down_scales[tp_part_idx][logical_expert_id]);
              return;
            }
            {
              size_t scale_size = config_.intermediate_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;

              memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
              }

              memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
              }
            }

            {
              size_t scale_size = config_.hidden_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;

              memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
              }
            }
          },
          nullptr);

    } else {
      int nth = T::recommended_nth(config_.intermediate_size);
      static uint8_t mat_type_all = 3, mat_split = 1;
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));

      if (config_.load) {
        std::cout << "Loading from \"" << prefix << "\"" << std::endl;
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all * mat_split,
            [this, physical_to_logical_map, prefix, mat_type_all, mat_split](int task_id) {
              int64_t expert_idx = task_id / (mat_type_all * mat_split);
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
              uint8_t mat_split_idex = task_id % mat_split;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                read_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              }
            });
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else
#endif
      {
        if (tp_part_idx == 0) {
          std::cout << "  online quant from bf16" << std::endl;
        }
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // gate part
              gate_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
              // up part
              up_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // down part
              down_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.down_proj + logical_expert_id * config_.hidden_size * config_.intermediate_size,
                  ith, nth);
              // printf("load idown, expert %ld, ith %d, total nth %d\n", expert_idx, ith, nth);
            },
            nullptr);
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this, physical_to_logical_map, prefix](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              expert_idx = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size);
              }
            },
            nullptr);
      }
    }
  }

  // forward, forward_prefill, forward_decode, warm_up are inherited from Base
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation
// ============================================================================

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                             col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      DO_TPS_LOAD_WEIGHTS(pool);

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file %s\n", config.path.c_str());
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("AMXINT4 BF16 export pointer arrays must match gpu_tp_count");
    }
    // AMXINT4 is dequantized to BF16, so scale destinations are intentionally
    // unused.  Reject partially populated arrays because that usually means a
    // caller selected an incompatible quantized GPU shadow layout.
    if ((!w13_scale_ptrs.empty() &&
         std::any_of(w13_scale_ptrs.begin(), w13_scale_ptrs.end(), [](uintptr_t ptr) { return ptr != 0; })) ||
        (!w2_scale_ptrs.empty() &&
         std::any_of(w2_scale_ptrs.begin(), w2_scale_ptrs.end(), [](uintptr_t ptr) { return ptr != 0; }))) {
      throw std::runtime_error("AMXINT4 BF16 export does not accept GPU scale destinations");
    }
    if (gpu_tp_count != this->tp_count && gpu_tp_count != 1) {
      throw std::runtime_error("AMXINT4 BF16 export requires GPU TP to match CPU NUMA/TP count or equal 1");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_bf16_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                                 w2_weight_ptrs);
    });
  }

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>
};

#endif
