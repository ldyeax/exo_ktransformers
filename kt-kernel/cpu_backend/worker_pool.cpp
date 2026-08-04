/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "worker_pool.h"

#include <hwloc/bitmap.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hwloc.h"

thread_local int WorkerPool::thread_local_id = -1;

namespace {

std::chrono::microseconds worker_spin_duration() {
  static const std::chrono::microseconds duration = [] {
    constexpr long long default_spin_us = 50000;
    const char* value = std::getenv("KT_WORKER_SPIN_US");
    if (value == nullptr) {
      return std::chrono::microseconds(default_spin_us);
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0) {
      std::fprintf(stderr,
                   "Ignoring invalid KT_WORKER_SPIN_US=%s; using %lld us\n",
                   value, default_spin_us);
      return std::chrono::microseconds(default_spin_us);
    }
    return std::chrono::microseconds(parsed);
  }();
  return duration;
}

bool single_numa_inline_dispatch_environment_requested() {
  const char* value = std::getenv("KT_SINGLE_NUMA_INLINE_DISPATCH");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

bool is_single_numa_subpool_config(const WorkerPoolConfig& config) {
  return config.subpool_count == 1 && config.subpool_numa_map.size() == 1 && config.subpool_thread_count.size() == 1 &&
         config.subpool_thread_count[0] > 0;
}

hwloc_obj_t numa_object_by_os_index(hwloc_topology_t topology, int numa_id) {
  hwloc_obj_t object = nullptr;
  while ((object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, object)) != nullptr) {
    if (static_cast<int>(object->os_index) == numa_id) return object;
  }
  return nullptr;
}

int physical_core_count(hwloc_topology_t topology, hwloc_obj_t numa_object) {
  return hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE);
}

// Preserve the existing worker-to-core mapping for the first hardware thread
// of every physical core.  Oversubscribed workers then consume the second PU
// of each core, followed by the third, and so on.  This keeps worker 56 on the
// sibling of core 0 on a 56-core SMT2 socket instead of wrapping onto CPU 0.
hwloc_obj_t processing_unit_for_worker_index(hwloc_topology_t topology, hwloc_obj_t numa_object, int worker_index) {
  const int core_count = physical_core_count(topology, numa_object);
  if (worker_index < 0 || core_count <= 0) return nullptr;
  if (worker_index < core_count) {
    hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE, worker_index);
    return core == nullptr ? nullptr : hwloc_get_obj_inside_cpuset_by_type(topology, core->cpuset, HWLOC_OBJ_PU, 0);
  }

  int remaining = worker_index - core_count;
  for (int sibling_index = 1;; ++sibling_index) {
    bool found_any_at_level = false;
    for (int core_index = 0; core_index < core_count; ++core_index) {
      hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE, core_index);
      if (core == nullptr) continue;
      hwloc_obj_t processing_unit =
          hwloc_get_obj_inside_cpuset_by_type(topology, core->cpuset, HWLOC_OBJ_PU, sibling_index);
      if (processing_unit == nullptr) continue;
      found_any_at_level = true;
      if (remaining == 0) return processing_unit;
      --remaining;
    }
    if (!found_any_at_level) return nullptr;
  }
}

long native_thread_id() { return static_cast<long>(syscall(SYS_gettid)); }

}  // namespace

InNumaPool::InNumaPool(int max_thread_num) {
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  worker_native_thread_ids_ = std::unique_ptr<std::atomic<long>[]>(new std::atomic<long>[max_thread_num]);
  worker_affinity_cpu_ids_.assign(max_thread_num, -1);
  worker_affinity_statuses_.assign(max_thread_num, AffinityStatus::CALLER);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
    worker_native_thread_ids_[i].store(-1, std::memory_order_relaxed);
  }
  workers_.resize(total_worker_count);
  try {
    for (int i = 1; i < total_worker_count; i++) {
      worker_affinity_statuses_[i] = AffinityStatus::PENDING;
      workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, -1);
    }
    std::unique_lock<std::mutex> lock(worker_startup_mutex_);
    worker_startup_cv_.wait(lock, [&] { return started_background_worker_count_ == total_worker_count - 1; });
  } catch (...) {
    stop_and_join_workers();
    throw;
  }
}

InNumaPool::InNumaPool(int max_thread_num, int numa_id, int threads_id_start) {
  printf("===========In NumaPool============\n");
  if (max_thread_num <= 0) throw std::invalid_argument("InNumaPool thread count must be positive");
  hwloc_topology_t topology = nullptr;
  if (hwloc_topology_init(&topology) != 0 || hwloc_topology_load(topology) != 0) {
    if (topology != nullptr) hwloc_topology_destroy(topology);
    throw std::runtime_error("failed to load hwloc topology for InNumaPool");
  }
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  worker_native_thread_ids_ = std::unique_ptr<std::atomic<long>[]>(new std::atomic<long>[max_thread_num]);
  worker_affinity_cpu_ids_.assign(max_thread_num, -1);
  worker_affinity_statuses_.assign(max_thread_num, AffinityStatus::CALLER);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
    worker_native_thread_ids_[i].store(-1, std::memory_order_relaxed);
  }
  workers_.resize(total_worker_count);
  hwloc_obj_t numa_object = numa_object_by_os_index(topology, numa_id);
  if (numa_object == nullptr) {
    hwloc_topology_destroy(topology);
    throw std::invalid_argument("NUMA node not found for InNumaPool");
  }
  const int core_count = physical_core_count(topology, numa_object);

  try {
    for (int i = 1; i < total_worker_count; i++) {
      worker_affinity_statuses_[i] = AffinityStatus::PENDING;
      workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, numa_id);
      const std::string thread_name = "numa_" + std::to_string(numa_id) + "_t_" + std::to_string(i + threads_id_start);
      pthread_t native_handle = workers_[i].native_handle();
      const int set_name_result = pthread_setname_np(native_handle, thread_name.c_str());
      if (set_name_result != 0) {
        fprintf(stderr, "Failed to set thread name: %s\n", strerror(set_name_result));
      }

      const int worker_index = i + threads_id_start;
      hwloc_obj_t processing_unit = processing_unit_for_worker_index(topology, numa_object, worker_index);
      if (processing_unit == nullptr) {
        worker_affinity_statuses_[i] = AffinityStatus::PROCESSING_UNIT_NOT_FOUND;
      } else {
        hwloc_bitmap_t target_cpuset = hwloc_bitmap_dup(processing_unit->cpuset);
        hwloc_bitmap_t observed_cpuset = hwloc_bitmap_alloc();
        if (target_cpuset == nullptr || observed_cpuset == nullptr) {
          if (target_cpuset != nullptr) hwloc_bitmap_free(target_cpuset);
          if (observed_cpuset != nullptr) hwloc_bitmap_free(observed_cpuset);
          worker_affinity_statuses_[i] = AffinityStatus::CPUSET_ALLOCATION_FAILED;
        } else {
          hwloc_bitmap_singlify(target_cpuset);
          const int cpu_id = hwloc_bitmap_first(target_cpuset);
          if (hwloc_set_thread_cpubind(topology, native_handle, target_cpuset, HWLOC_CPUBIND_STRICT) != 0) {
            worker_affinity_statuses_[i] = AffinityStatus::BIND_FAILED;
          } else if (hwloc_get_thread_cpubind(topology, native_handle, observed_cpuset, HWLOC_CPUBIND_THREAD) != 0 ||
                     !hwloc_bitmap_isequal(target_cpuset, observed_cpuset)) {
            worker_affinity_statuses_[i] = AffinityStatus::VERIFY_FAILED;
          } else {
            worker_affinity_cpu_ids_[i] = cpu_id;
            worker_affinity_statuses_[i] = AffinityStatus::ACTIVE;
          }
          hwloc_bitmap_free(observed_cpuset);
          hwloc_bitmap_free(target_cpuset);
        }
      }

      if (worker_affinity_statuses_[i] != AffinityStatus::ACTIVE) {
        fprintf(stderr, "Failed to bind NUMA %d worker %d: %s\n", numa_id, worker_index,
                affinity_status_name(worker_affinity_statuses_[i]));
        // Existing physical-core workers retain the historical best-effort
        // behavior.  Extra workers must never float or wrap onto a primary PU.
        if (worker_index >= core_count) {
          throw std::runtime_error("failed to bind oversubscribed NUMA worker " + std::to_string(worker_index) + ": " +
                                   affinity_status_name(worker_affinity_statuses_[i]));
        }
      }
    }
    {
      std::unique_lock<std::mutex> lock(worker_startup_mutex_);
      worker_startup_cv_.wait(lock, [&] { return started_background_worker_count_ == total_worker_count - 1; });
    }
    hwloc_topology_destroy(topology);
  } catch (...) {
    hwloc_topology_destroy(topology);
    stop_and_join_workers();
    throw;
  }
}

InNumaPool::~InNumaPool() { stop_and_join_workers(); }

void InNumaPool::stop_and_join_workers() noexcept {
  for (int i = 0; i < total_worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::EXIT, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  for (int i = 0; i < total_worker_count; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
}

const char* InNumaPool::affinity_status_name(AffinityStatus status) {
  switch (status) {
    case AffinityStatus::CALLER:
      return "caller";
    case AffinityStatus::PENDING:
      return "pending";
    case AffinityStatus::ACTIVE:
      return "active";
    case AffinityStatus::NUMA_NODE_NOT_FOUND:
      return "numa_node_not_found";
    case AffinityStatus::PROCESSING_UNIT_NOT_FOUND:
      return "processing_unit_not_found";
    case AffinityStatus::CPUSET_ALLOCATION_FAILED:
      return "cpuset_allocation_failed";
    case AffinityStatus::BIND_FAILED:
      return "bind_failed";
    case AffinityStatus::VERIFY_FAILED:
      return "verify_failed";
  }
  return "unknown";
}

int InNumaPool::configured_worker_count() const { return total_worker_count; }

std::vector<int> InNumaPool::worker_affinity_cpu_ids() const { return worker_affinity_cpu_ids_; }

std::vector<long> InNumaPool::worker_native_thread_ids() const {
  std::vector<long> result(total_worker_count, -1);
  for (int i = 0; i < total_worker_count; ++i) {
    result[i] = worker_native_thread_ids_[i].load(std::memory_order_acquire);
  }
  return result;
}

std::vector<std::string> InNumaPool::worker_affinity_statuses() const {
  std::vector<std::string> result;
  result.reserve(worker_affinity_statuses_.size());
  for (AffinityStatus status : worker_affinity_statuses_) result.emplace_back(affinity_status_name(status));
  return result;
}

long InNumaPool::last_caller_native_thread_id() const {
  return last_caller_native_thread_id_.load(std::memory_order_acquire);
}

int InNumaPool::last_caller_cpu_id() const { return last_caller_cpu_id_.load(std::memory_order_acquire); }

int InNumaPool::get_thread_num() {
  throw std::runtime_error("Deprecated");
  return total_worker_count;
}

void InNumaPool::set_restricted_worker_count(int count) { restricted_worker_count = count; }

void InNumaPool::wait() {
  for (int i = 0; i < worker_count; i++) {
    while (thread_state_[i].status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }

#ifdef PROFILE_BALANCE
  size_t max_time = 0;
  size_t min_time = thread_state_[0].finish_ns;
  size_t sum = 0;
  for (int i = 0; i < worker_count; i++) {
    sum += thread_state_[i].finish_ns;
    max_time = std::max(max_time, thread_state_[i].finish_ns);
    min_time = std::min(min_time, thread_state_[i].finish_ns);
  }
  double balance = 1.0 * sum / (max_time * worker_count);
  printf("max_time: %ld, min_time: %ld, sum_time: %ld, balance: %f\n", max_time, min_time, sum, balance);

#endif
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);
  wait();
}

void InNumaPool::do_work_stealing_job_async(int task_num, std::function<void(int)> init_func,
                                            std::function<void(int)> compute_func,
                                            std::function<void(int)> finalize_func) {
  init_func_ = init_func;
  compute_func_ = compute_func;
  finalize_func_ = finalize_func;
  worker_count = std::min(restricted_worker_count, task_num);
  curr_.store(0, std::memory_order_release);
  end_ = task_num;
  for (int i = 0; i < worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::WORKING, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  last_caller_native_thread_id_.store(native_thread_id(), std::memory_order_release);
  last_caller_cpu_id_.store(sched_getcpu(), std::memory_order_release);
  worker_native_thread_ids_[0].store(last_caller_native_thread_id_.load(std::memory_order_relaxed),
                                     std::memory_order_release);
  WorkerPool::thread_local_id = 0;
  process_tasks(0);
}

void InNumaPool::process_tasks(int thread_id) {
#ifdef PROFILE_BALANCE
  auto start = std::chrono::high_resolution_clock::now();
#endif
  auto& s = thread_state_[thread_id];
  if (init_func_ != nullptr) {
    init_func_(thread_id);
  }

  // omp-guided-style work scheduling
  while (true) {
    int old = curr_.load(std::memory_order_relaxed);
    int rem = end_ - old;
    if (rem <= 0) {
      break;
    }

    int block = (rem + worker_count - 1) / worker_count;
    block = 1;
    int task_id = curr_.fetch_add(block, std::memory_order_acq_rel);
    if (task_id >= end_) {
      break;
    }

    for (int i = 0; i < block; i++) {
      if (task_id + i >= end_) {
        break;
      }
      compute_func_(task_id + i);
    }
  }

  if (finalize_func_ != nullptr) {
    finalize_func_(thread_id);
  }

  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
#ifdef PROFILE_BALANCE
  s.finish_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#endif
}

void InNumaPool::worker_thread(int thread_id, int numa_id) {
  worker_native_thread_ids_[thread_id].store(native_thread_id(), std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(worker_startup_mutex_);
    ++started_background_worker_count_;
  }
  worker_startup_cv_.notify_one();
  if (numa_id >= 0) {
    set_memory_to_numa(numa_id);
  }
  auto start = std::chrono::high_resolution_clock::now();
  const auto spin_duration = worker_spin_duration();
  WorkerPool::thread_local_id = thread_id;  // 设置线程本地变量
  while (true) {
    ThreadStatus status = thread_state_[thread_id].status.load(std::memory_order_acquire);
    if (status == ThreadStatus::WORKING) {
      process_tasks(thread_id);
      start = std::chrono::high_resolution_clock::now();
    } else if (status == ThreadStatus::WAITING) {
      const bool should_sleep =
          spin_duration.count() == 0 ||
          std::chrono::high_resolution_clock::now() - start >= spin_duration;
      if (should_sleep) {
        std::unique_lock<std::mutex> lock(thread_state_[thread_id].mutex);
        thread_state_[thread_id].cv.wait(lock, [&] {
          return thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (status == ThreadStatus::EXIT) {
      return;
    }
  }
}

NumaJobDistributor::NumaJobDistributor(int numa_count) {
  std::vector<int> numa_ids;
  for (int i = 0; i < numa_count; i++) {
    numa_ids.push_back(i);
  }
  init(numa_ids);
}

NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids) { init(numa_ids); }
NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count) {
  init(numa_ids, thread_count, false);
}
NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count,
                                       bool single_numa_inline_dispatch) {
  init(numa_ids, thread_count, single_numa_inline_dispatch);
}

void NumaJobDistributor::init(std::vector<int> numa_ids) {
  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (int i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  for (int i = 0; i < numa_count; i++) {
    std::thread([this, i]() { workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i); }).join();
  }
  ready_bar->arrive_and_wait();
}

void NumaJobDistributor::init(std::vector<int> numa_ids, std::vector<int> thread_count,
                              bool single_numa_inline_dispatch) {
  this->numa_count = numa_ids.size();
  physical_numa_id_ = numa_ids.size() == 1 ? numa_ids[0] : -1;
  if (single_numa_inline_dispatch) {
    if (numa_ids.size() != 1 || thread_count.size() != 1 || thread_count[0] <= 0) {
      throw std::invalid_argument("single-NUMA inline dispatch requires exactly one non-empty subpool");
    }
    this->numa_ids = std::move(numa_ids);
    single_numa_inline_dispatch_active_ = true;
    return;
  }

  hwloc_topology_t topology = nullptr;
  if (hwloc_topology_init(&topology) != 0 || hwloc_topology_load(topology) != 0) {
    if (topology != nullptr) hwloc_topology_destroy(topology);
    throw std::runtime_error("failed to load hwloc topology for NumaJobDistributor");
  }

  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (int i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  // numa_ids contains physical NUMA IDs, which need not be dense within the
  // selected subpool list.  In particular, a one-subpool configuration for
  // physical node 1 has numa_count == 1 but still indexes ID 1.
  // Indexing a numa_count-sized vector here read past the end and produced a
  // bogus start_id (observed as 32768), silently defeating strict core
  // affinity for nearly every worker.
  std::vector<int> numa_threads_count(numa_num_configured_nodes(), 0);
  for (int i = 0; i < numa_count; i++) {
    auto this_numa = numa_ids[i];
    if (this_numa < 0 || this_numa >= static_cast<int>(numa_threads_count.size())) {
      throw std::invalid_argument("NUMA ID outside configured node range");
    }
    workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i);
    auto start_id = numa_threads_count[this_numa];
    // set the thread name as: "worker_numa_(numa_id)_main_start_id(0)"
    // printf("nuam_id %d, start_id %d\n", this_numa, start_id);
    std::string thread_name = "numa_" + std::to_string(numa_ids[i]) + "_m_" + std::to_string(start_id);
    pthread_t native_handle = workers[i].native_handle();
    pthread_setname_np(native_handle, thread_name.c_str());
    // Set the thread affinity to the specified NUMA node's CPU (0)
    hwloc_obj_t numa_object = numa_object_by_os_index(topology, this_numa);
    if (!numa_object) {
      fprintf(stderr, "NUMA node %d not found\n", this_numa);
      // throw std::runtime_error("NUMA node not found");
      continue;
    }
    hwloc_obj_t processing_unit = processing_unit_for_worker_index(topology, numa_object, start_id);
    if (!processing_unit) {
      fprintf(stderr, "Processing unit for worker %d inside NUMA node %d not found\n", start_id, this_numa);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    // 精简 cpuset
    auto cpuset_simple = hwloc_bitmap_dup(processing_unit->cpuset);
    hwloc_bitmap_singlify(cpuset_simple);
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset_simple, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    }
    // 检查线程是否绑定到指定的 核上了
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_THREAD);
    // hwloc_bitmap_foreach_begin(i_in, cpuset) { printf("Thread %d is bound to CPU %ld\n", start_id, i_in); }
    // hwloc_bitmap_foreach_end();

    numa_threads_count[this_numa] += thread_count[i];
    hwloc_bitmap_free(cpuset);
    hwloc_bitmap_free(cpuset_simple);
  }
  ready_bar->arrive_and_wait();
  hwloc_topology_destroy(topology);
}

NumaJobDistributor::~NumaJobDistributor() {
  for (size_t i = 0; i < workers.size(); i++) {
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::EXIT, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (size_t i = 0; i < workers.size(); i++) {
    if (workers[i].joinable()) {
      workers[i].join();
    }
  }
}

bool NumaJobDistributor::single_numa_inline_dispatch_active() const { return single_numa_inline_dispatch_active_; }

int NumaJobDistributor::physical_numa_id() const { return physical_numa_id_; }

size_t NumaJobDistributor::distributor_worker_count() const { return workers.size(); }

unsigned long long NumaJobDistributor::inline_dispatch_count() const {
  return inline_dispatch_count_.load(std::memory_order_acquire);
}

unsigned long long NumaJobDistributor::inline_exception_count() const {
  return inline_exception_count_.load(std::memory_order_acquire);
}

long NumaJobDistributor::last_inline_native_thread_id() const {
  return last_inline_native_thread_id_.load(std::memory_order_acquire);
}

int NumaJobDistributor::last_inline_cpu_id() const { return last_inline_cpu_id_.load(std::memory_order_acquire); }

int NumaJobDistributor::last_inline_worker_pool_thread_id() const {
  return last_inline_worker_pool_thread_id_.load(std::memory_order_acquire);
}

#ifdef USE_NUMA_JOB_DIRECT_WORK

void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  if (single_numa_inline_dispatch_active_) {
    inline_dispatch_count_.fetch_add(1, std::memory_order_acq_rel);
    last_inline_native_thread_id_.store(native_thread_id(), std::memory_order_release);
    last_inline_cpu_id_.store(sched_getcpu(), std::memory_order_release);
    try {
      compute_func(0);
      last_inline_worker_pool_thread_id_.store(WorkerPool::thread_local_id, std::memory_order_release);
    } catch (...) {
      last_inline_worker_pool_thread_id_.store(WorkerPool::thread_local_id, std::memory_order_release);
      inline_exception_count_.fetch_add(1, std::memory_order_acq_rel);
      throw;
    }
    return;
  }
  this->compute_func = compute_func;
  auto me_numa = numa_node_of_cpu(sched_getcpu());
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  compute_func(me_numa);
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#else
void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  if (single_numa_inline_dispatch_active_) {
    inline_dispatch_count_.fetch_add(1, std::memory_order_acq_rel);
    last_inline_native_thread_id_.store(native_thread_id(), std::memory_order_release);
    last_inline_cpu_id_.store(sched_getcpu(), std::memory_order_release);
    try {
      compute_func(0);
      last_inline_worker_pool_thread_id_.store(WorkerPool::thread_local_id, std::memory_order_release);
    } catch (...) {
      last_inline_worker_pool_thread_id_.store(WorkerPool::thread_local_id, std::memory_order_release);
      inline_exception_count_.fetch_add(1, std::memory_order_acq_rel);
      throw;
    }
    return;
  }
  this->compute_func = compute_func;
  for (int i = 0; i < numa_count; i++) {
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#endif

void NumaJobDistributor::worker_thread(int numa_id) {
  auto start = std::chrono::high_resolution_clock::now();
  const auto spin_duration = worker_spin_duration();
  set_memory_to_numa(numa_ids[numa_id]);
  status[numa_id] =
      std::move(std::unique_ptr<std::atomic<ThreadStatus>>(new std::atomic<ThreadStatus>(ThreadStatus::WAITING)));
  ready_bar->arrive_and_wait();
  while (true) {
    auto stat = status[numa_id]->load(std::memory_order_acquire);
    if (stat == ThreadStatus::WORKING) {
      compute_func(numa_id);
      status[numa_id]->store(ThreadStatus::WAITING, std::memory_order_release);
      start = std::chrono::high_resolution_clock::now();
    } else if (stat == ThreadStatus::WAITING) {
      const bool should_sleep =
          spin_duration.count() == 0 ||
          std::chrono::high_resolution_clock::now() - start >= spin_duration;
      if (should_sleep) {
        std::unique_lock<std::mutex> lock(*mutexes[numa_id]);
        cvs[numa_id]->wait(lock, [&] {
          return status[numa_id]->load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (stat == ThreadStatus::EXIT) {
      return;
    }
  }
}

void WorkerPool::init(WorkerPoolConfig config) {
  single_numa_inline_dispatch_environment_enabled_ = single_numa_inline_dispatch_environment_requested();
  single_numa_inline_dispatch_eligible_ = is_single_numa_subpool_config(config);
  if (single_numa_inline_dispatch_environment_enabled_ && !single_numa_inline_dispatch_eligible_) {
    throw std::invalid_argument(
        "KT_SINGLE_NUMA_INLINE_DISPATCH=1 requires exactly one physical NUMA subpool with a positive thread count");
  }
  if (config.subpool_count <= 0 || config.subpool_numa_map.size() != static_cast<size_t>(config.subpool_count) ||
      config.subpool_thread_count.size() != static_cast<size_t>(config.subpool_count)) {
    throw std::invalid_argument("WorkerPoolConfig subpool vectors do not match subpool_count");
  }

  printf("WorkerPool[0x%lx] %d subpools, [numa:threads]", (intptr_t)this, config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    printf("[%d:%d] ", config.subpool_numa_map[i], config.subpool_thread_count[i]);
  }
  printf("\n");

  numa_count = config.subpool_count;
  total_thread_count = 0;
  threads_per_numa = config.subpool_thread_count[0];
  numa_worker_pools.resize(config.subpool_count);
  // subpool_numa_map stores physical NUMA IDs.  Size this accounting table
  // by the machine topology rather than the number of selected subpools so
  // an explicit single-node-1 pool cannot index out of bounds.
  std::vector<int> numa_threads_count(numa_num_configured_nodes(), 0);
  for (int i = 0; i < config.subpool_count; ++i) {
    const int physical_numa_id = config.subpool_numa_map[i];
    if (physical_numa_id < 0 || physical_numa_id >= static_cast<int>(numa_threads_count.size())) {
      throw std::invalid_argument("NUMA ID outside configured node range");
    }
    if (config.subpool_thread_count[i] <= 0) {
      throw std::invalid_argument("WorkerPoolConfig thread counts must be positive");
    }
    total_thread_count += config.subpool_thread_count[i];
  }
  for (int i = 0; i < config.subpool_count; i++) {
    auto this_numa = config.subpool_numa_map[i];
    auto this_thread_count = config.subpool_thread_count[i];
    auto this_thread_id_start = numa_threads_count[this_numa];
    std::exception_ptr construction_error;
    std::thread construction_thread(
        [this, i, this_numa, this_thread_count, this_thread_id_start, &construction_error]() {
          try {
            set_to_numa(this_numa);
            numa_worker_pools[i] = std::make_unique<InNumaPool>(this_thread_count, this_numa, this_thread_id_start);
          } catch (...) {
            construction_error = std::current_exception();
          }
        });
    construction_thread.join();
    if (construction_error != nullptr) std::rethrow_exception(construction_error);
    if (single_numa_inline_dispatch_environment_enabled_) {
      const std::vector<std::string> affinity_statuses = numa_worker_pools[i]->worker_affinity_statuses();
      const auto failed_binding = std::find_if(affinity_statuses.begin() + 1, affinity_statuses.end(),
                                               [](const std::string& status) { return status != "active"; });
      if (failed_binding != affinity_statuses.end()) {
        const int logical_worker_id = static_cast<int>(failed_binding - affinity_statuses.begin());
        numa_worker_pools[i].reset();
        throw std::runtime_error("single-NUMA inline dispatch requires strict affinity for logical worker " +
                                 std::to_string(logical_worker_id) + ": " + *failed_binding);
      }
    }
    numa_threads_count[this_numa] += this_thread_count;
  }

  distributor = std::make_unique<NumaJobDistributor>(
      config.subpool_numa_map, config.subpool_thread_count,
      single_numa_inline_dispatch_environment_enabled_ && single_numa_inline_dispatch_eligible_);
  // distributor = std::move(std::unique_ptr<NumaJobDistributor>(new NumaJobDistributor(config.subpool_numa_map)));
}

WorkerPool::WorkerPool(WorkerPoolConfig config) : config(config) { init(config); }

WorkerPool::WorkerPool(int total_threads) {
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = i;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::WorkerPool(int total_threads, int single_numa_id) {
  set_to_numa(single_numa_id);
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = single_numa_id;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::~WorkerPool() {}

int WorkerPool::get_thread_num() { return total_thread_count; }

void WorkerPool::set_restricted_worker_count(int count) {
  (void)count;
  for (int i = 0; i < numa_count; i++) {
    numa_worker_pools[i]->set_restricted_worker_count(threads_per_numa);
  }
}

InNumaPool* WorkerPool::get_subpool(int numa_id) { return numa_worker_pools[numa_id].get(); }

NumaJobDistributor* WorkerPool::dispense_backend() { return distributor.get(); }

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  numa_worker_pools[0]->do_work_stealing_job(task_num, init_func, compute_func, finalize_func);
}

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}

bool WorkerPool::single_numa_inline_dispatch_environment_enabled() const {
  return single_numa_inline_dispatch_environment_enabled_;
}

bool WorkerPool::single_numa_inline_dispatch_eligible() const { return single_numa_inline_dispatch_eligible_; }

bool WorkerPool::single_numa_inline_dispatch_requested() const {
  return single_numa_inline_dispatch_environment_enabled_;
}

bool WorkerPool::single_numa_inline_dispatch_active() const {
  return distributor != nullptr && distributor->single_numa_inline_dispatch_active();
}

const char* WorkerPool::single_numa_inline_dispatch_status() const {
  if (!single_numa_inline_dispatch_environment_enabled_) return "not_requested";
  if (!single_numa_inline_dispatch_eligible_) return "ineligible";
  return single_numa_inline_dispatch_active() ? "active" : "inactive";
}

int WorkerPool::single_numa_inline_dispatch_physical_numa_id() const {
  return distributor == nullptr ? -1 : distributor->physical_numa_id();
}

size_t WorkerPool::single_numa_inline_dispatch_worker_count() const {
  return distributor == nullptr ? 0 : distributor->distributor_worker_count();
}

unsigned long long WorkerPool::single_numa_inline_dispatch_count() const {
  return distributor == nullptr ? 0 : distributor->inline_dispatch_count();
}

unsigned long long WorkerPool::single_numa_inline_exception_count() const {
  return distributor == nullptr ? 0 : distributor->inline_exception_count();
}

long WorkerPool::single_numa_inline_dispatch_last_native_thread_id() const {
  return distributor == nullptr ? -1 : distributor->last_inline_native_thread_id();
}

int WorkerPool::single_numa_inline_dispatch_last_cpu_id() const {
  return distributor == nullptr ? -1 : distributor->last_inline_cpu_id();
}

int WorkerPool::single_numa_inline_dispatch_last_worker_pool_thread_id() const {
  return distributor == nullptr ? -1 : distributor->last_inline_worker_pool_thread_id();
}

std::vector<int> WorkerPool::subpool_worker_affinity_cpu_ids(int subpool_index) const {
  if (subpool_index < 0 || subpool_index >= static_cast<int>(numa_worker_pools.size())) {
    throw std::out_of_range("subpool index outside WorkerPool");
  }
  return numa_worker_pools[subpool_index]->worker_affinity_cpu_ids();
}

std::vector<long> WorkerPool::subpool_worker_native_thread_ids(int subpool_index) const {
  if (subpool_index < 0 || subpool_index >= static_cast<int>(numa_worker_pools.size())) {
    throw std::out_of_range("subpool index outside WorkerPool");
  }
  return numa_worker_pools[subpool_index]->worker_native_thread_ids();
}

std::vector<std::string> WorkerPool::subpool_worker_affinity_statuses(int subpool_index) const {
  if (subpool_index < 0 || subpool_index >= static_cast<int>(numa_worker_pools.size())) {
    throw std::out_of_range("subpool index outside WorkerPool");
  }
  return numa_worker_pools[subpool_index]->worker_affinity_statuses();
}

long WorkerPool::subpool_last_caller_native_thread_id(int subpool_index) const {
  if (subpool_index < 0 || subpool_index >= static_cast<int>(numa_worker_pools.size())) {
    throw std::out_of_range("subpool index outside WorkerPool");
  }
  return numa_worker_pools[subpool_index]->last_caller_native_thread_id();
}

int WorkerPool::subpool_last_caller_cpu_id(int subpool_index) const {
  if (subpool_index < 0 || subpool_index >= static_cast<int>(numa_worker_pools.size())) {
    throw std::out_of_range("subpool index outside WorkerPool");
  }
  return numa_worker_pools[subpool_index]->last_caller_cpu_id();
}
