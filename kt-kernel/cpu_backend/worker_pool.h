/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <hwloc.h>
#include <numa.h>

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// #define PROFILE_BALANCE

inline void set_to_numa(int this_numa) {
  struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
  numa_bitmask_setbit(mask, this_numa);
  numa_bind(mask);
  numa_bitmask_free(mask);
}

inline void set_memory_to_numa(int this_numa) {
  // printf("Set memory to NUMA %d\n", this_numa);
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, this_numa);
  if (!obj) {
    fprintf(stderr, "NUMA node %d not found.\n", this_numa);
    hwloc_topology_destroy(topology);
    return;
  }

  auto ret = hwloc_set_membind(topology, obj->nodeset, HWLOC_MEMBIND_BIND,
                               HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);
  if (ret != 0) {
    perror("hwloc_set_membind_nodeset");
  }

  hwloc_topology_destroy(topology);
}

enum ThreadStatus {
  WORKING,
  WAITING,
  EXIT,
};

struct alignas(64) ThreadState {
  std::atomic<ThreadStatus> status;
  std::mutex mutex;
  std::condition_variable cv;
#ifdef PROFILE_BALANCE
  size_t finish_ns;
#endif
};

class InNumaPool {
 public:
  enum class AffinityStatus {
    CALLER,
    PENDING,
    ACTIVE,
    NUMA_NODE_NOT_FOUND,
    PROCESSING_UNIT_NOT_FOUND,
    CPUSET_ALLOCATION_FAILED,
    BIND_FAILED,
    VERIFY_FAILED,
  };

  InNumaPool(int thread_count);
  InNumaPool(int max_thread_num, int numa_id, int threads_id_start);
  ~InNumaPool();
  int get_thread_num();
  void set_restricted_worker_count(int count);

  void do_work_stealing_job_async(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);
  void wait();

  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);
  void do_work_stealing_job(int, std::function<void(int)>);

  int configured_worker_count() const;
  std::vector<int> worker_affinity_cpu_ids() const;
  std::vector<long> worker_native_thread_ids() const;
  std::vector<std::string> worker_affinity_statuses() const;
  long last_caller_native_thread_id() const;
  int last_caller_cpu_id() const;

 private:
  int worker_count;
  int total_worker_count;

  std::unique_ptr<ThreadState[]> thread_state_;  // [thread_num]
  std::vector<std::thread> workers_;

  // changed ever time called do_work_stealing_job_async
  int restricted_worker_count;
  std::function<void(int)> init_func_;
  std::function<void(int)> compute_func_;
  std::function<void(int)> finalize_func_;
  std::atomic<int> curr_;
  int end_;

  std::vector<int> worker_affinity_cpu_ids_;
  std::vector<AffinityStatus> worker_affinity_statuses_;
  std::unique_ptr<std::atomic<long>[]> worker_native_thread_ids_;
  std::atomic<long> last_caller_native_thread_id_{-1};
  std::atomic<int> last_caller_cpu_id_{-1};
  std::mutex worker_startup_mutex_;
  std::condition_variable worker_startup_cv_;
  int started_background_worker_count_ = 0;

  void process_tasks(int);
  void worker_thread(int, int);
  void stop_and_join_workers() noexcept;
  static const char* affinity_status_name(AffinityStatus);
};

class NumaJobDistributor {
 public:
  NumaJobDistributor(int numa_count);
  NumaJobDistributor(std::vector<int> numa_ids);
  NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count);
  NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count, bool single_numa_inline_dispatch);

  ~NumaJobDistributor();

  void do_numa_job(std::function<void(int)>);

  bool single_numa_inline_dispatch_active() const;
  int physical_numa_id() const;
  size_t distributor_worker_count() const;
  unsigned long long inline_dispatch_count() const;
  unsigned long long inline_exception_count() const;
  long last_inline_native_thread_id() const;
  int last_inline_cpu_id() const;
  int last_inline_worker_pool_thread_id() const;

 private:
  void init(std::vector<int> numa_ids);
  void init(std::vector<int> numa_ids, std::vector<int> thread_count, bool single_numa_inline_dispatch = false);

  std::unique_ptr<std::barrier<>> ready_bar;

  int numa_count;
  std::vector<int> numa_ids;
  std::vector<std::unique_ptr<std::atomic<ThreadStatus>>> status;
  std::vector<std::unique_ptr<std::mutex>> mutexes;
  std::vector<std::unique_ptr<std::condition_variable>> cvs;
  std::function<void(int)> compute_func;
  std::vector<std::thread> workers;

  bool single_numa_inline_dispatch_active_ = false;
  int physical_numa_id_ = -1;
  std::atomic<unsigned long long> inline_dispatch_count_{0};
  std::atomic<unsigned long long> inline_exception_count_{0};
  std::atomic<long> last_inline_native_thread_id_{-1};
  std::atomic<int> last_inline_cpu_id_{-1};
  std::atomic<int> last_inline_worker_pool_thread_id_{-1};

  void worker_thread(int);
};

struct WorkerPoolConfig {
  int subpool_count;
  std::vector<int> subpool_numa_map;
  std::vector<int> subpool_thread_count;
};

class WorkerPool {
 public:
  WorkerPool(int total_thread_count);
  WorkerPool(int total_thread_count, int single_numa_id);
  WorkerPool(WorkerPoolConfig config);
  ~WorkerPool();
  int get_thread_num();
  void set_restricted_worker_count(int count);

  static thread_local int thread_local_id;

  NumaJobDistributor* dispense_backend();

  InNumaPool* get_subpool(int numa_id);

  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);
  void do_work_stealing_job(int, std::function<void(int)>);

  bool single_numa_inline_dispatch_environment_enabled() const;
  bool single_numa_inline_dispatch_eligible() const;
  bool single_numa_inline_dispatch_requested() const;
  bool single_numa_inline_dispatch_active() const;
  const char* single_numa_inline_dispatch_status() const;
  int single_numa_inline_dispatch_physical_numa_id() const;
  size_t single_numa_inline_dispatch_worker_count() const;
  unsigned long long single_numa_inline_dispatch_count() const;
  unsigned long long single_numa_inline_exception_count() const;
  long single_numa_inline_dispatch_last_native_thread_id() const;
  int single_numa_inline_dispatch_last_cpu_id() const;
  int single_numa_inline_dispatch_last_worker_pool_thread_id() const;
  std::vector<int> subpool_worker_affinity_cpu_ids(int subpool_index) const;
  std::vector<long> subpool_worker_native_thread_ids(int subpool_index) const;
  std::vector<std::string> subpool_worker_affinity_statuses(int subpool_index) const;
  long subpool_last_caller_native_thread_id(int subpool_index) const;
  int subpool_last_caller_cpu_id(int subpool_index) const;

  WorkerPoolConfig config;

 private:
  void init(WorkerPoolConfig config);

  int total_thread_count;
  int numa_count;
  int threads_per_numa;
  std::unique_ptr<NumaJobDistributor> distributor;

  bool single_numa_inline_dispatch_environment_enabled_ = false;
  bool single_numa_inline_dispatch_eligible_ = false;

  std::vector<std::unique_ptr<InNumaPool>> numa_worker_pools;
};

#endif
