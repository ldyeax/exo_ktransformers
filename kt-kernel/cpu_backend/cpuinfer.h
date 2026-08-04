/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#ifdef KTRANSFORMERS_USE_CUDA
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#endif

#include "./vendors/vendor.h"
#include "llama.cpp/ggml/include/ggml-cpu.h"
#include "llama.cpp/ggml/src/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

inline bool task_queue_pin_first_core_environment_enabled() {
  const char* value = std::getenv("KT_TASK_QUEUE_PIN_FIRST_CORE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    initialize_task_queue(false, -1);
    ggml_cpu_init();
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    initialize_task_queue(false, -1);
    ggml_cpu_init();
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    task_queue_affinity_eligible_ =
        config.subpool_count == 1 && config.subpool_numa_map.size() == 1 && config.subpool_thread_count.size() == 1;
    backend_ = new WorkerPool(config);
    // Inline dispatch makes the TaskQueue thread the InNumaPool logical
    // worker 0.  Pinning it to the reserved primary PU is therefore part of
    // the inline contract, even when the standalone affinity experiment's
    // environment variable is not set.
    task_queue_affinity_environment_enabled_ =
        task_queue_affinity_environment_enabled_ || backend_->single_numa_inline_dispatch_active();
    initialize_task_queue(task_queue_affinity_eligible_,
                          task_queue_affinity_eligible_ ? config.subpool_numa_map[0] : -1);
    ggml_cpu_init();
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    // TaskQueue closures may still be executing against backend_.  Drain and
    // join the queue before releasing the worker pools they reference.
    delete task_queue_;
    delete backend_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    sync_(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif

  bool task_queue_affinity_environment_enabled() const { return task_queue_affinity_environment_enabled_; }
  bool task_queue_affinity_eligible() const { return task_queue_affinity_eligible_; }
  bool task_queue_affinity_requested() const { return task_queue_->affinity_requested(); }
  bool task_queue_affinity_active() const { return task_queue_->affinity_active(); }
  int task_queue_affinity_numa_id() const { return task_queue_->affinity_numa_id(); }
  int task_queue_affinity_cpu_id() const { return task_queue_->affinity_cpu_id(); }
  long task_queue_affinity_native_thread_id() const { return task_queue_->affinity_native_thread_id(); }
  const char* task_queue_affinity_status() const { return task_queue_->affinity_status(); }

  bool single_numa_inline_dispatch_environment_enabled() const {
    return backend_->single_numa_inline_dispatch_environment_enabled();
  }
  bool single_numa_inline_dispatch_eligible() const { return backend_->single_numa_inline_dispatch_eligible(); }
  bool single_numa_inline_dispatch_requested() const { return backend_->single_numa_inline_dispatch_requested(); }
  bool single_numa_inline_dispatch_active() const { return backend_->single_numa_inline_dispatch_active(); }
  const char* single_numa_inline_dispatch_status() const { return backend_->single_numa_inline_dispatch_status(); }
  int single_numa_inline_dispatch_physical_numa_id() const {
    return backend_->single_numa_inline_dispatch_physical_numa_id();
  }
  size_t single_numa_inline_dispatch_worker_count() const {
    return backend_->single_numa_inline_dispatch_worker_count();
  }
  unsigned long long single_numa_inline_dispatch_count() const { return backend_->single_numa_inline_dispatch_count(); }
  unsigned long long single_numa_inline_dispatch_exception_count() const {
    return backend_->single_numa_inline_exception_count();
  }
  long single_numa_inline_dispatch_last_native_thread_id() const {
    return backend_->single_numa_inline_dispatch_last_native_thread_id();
  }
  int single_numa_inline_dispatch_last_cpu_id() const { return backend_->single_numa_inline_dispatch_last_cpu_id(); }
  int single_numa_inline_dispatch_last_worker_pool_thread_id() const {
    return backend_->single_numa_inline_dispatch_last_worker_pool_thread_id();
  }

 public:
  WorkerPool* backend_ = nullptr;
  TaskQueue* task_queue_ = nullptr;

 private:
  bool task_queue_affinity_environment_enabled_ = task_queue_pin_first_core_environment_enabled();
  bool task_queue_affinity_eligible_ = false;

  void initialize_task_queue(bool eligible, int numa_id) {
    try {
      if (task_queue_affinity_environment_enabled_ && eligible) {
        task_queue_ = new TaskQueue(numa_id);
      } else {
        task_queue_ = new TaskQueue();
      }
    } catch (...) {
      delete backend_;
      backend_ = nullptr;
      throw;
    }
  }
};

#endif
