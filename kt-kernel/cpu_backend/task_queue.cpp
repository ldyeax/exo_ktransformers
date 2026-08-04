/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <hwloc.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

struct AffinityResult {
  bool active;
  int error_number;
  int cpu_id;
};

TaskQueue::AffinityStatus bind_to_first_core(int numa_id, AffinityResult* result) {
  hwloc_topology_t topology = nullptr;
  if (hwloc_topology_init(&topology) != 0) {
    result->error_number = errno;
    return TaskQueue::AffinityStatus::TOPOLOGY_INIT_FAILED;
  }
  if (hwloc_topology_load(topology) != 0) {
    result->error_number = errno;
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::TOPOLOGY_LOAD_FAILED;
  }

  hwloc_obj_t numa_object = nullptr;
  while ((numa_object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_object)) != nullptr) {
    if (static_cast<int>(numa_object->os_index) == numa_id) {
      break;
    }
  }
  if (numa_object == nullptr) {
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::NUMA_NODE_NOT_FOUND;
  }

  hwloc_obj_t core_object = hwloc_get_obj_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE, 0);
  if (core_object == nullptr) {
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::CORE_NOT_FOUND;
  }

  hwloc_bitmap_t target_cpuset = hwloc_bitmap_dup(core_object->cpuset);
  hwloc_bitmap_t observed_cpuset = hwloc_bitmap_alloc();
  if (target_cpuset == nullptr || observed_cpuset == nullptr) {
    if (target_cpuset != nullptr) hwloc_bitmap_free(target_cpuset);
    if (observed_cpuset != nullptr) hwloc_bitmap_free(observed_cpuset);
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::CPUSET_ALLOCATION_FAILED;
  }
  hwloc_bitmap_singlify(target_cpuset);
  const int cpu_id = hwloc_bitmap_first(target_cpuset);
  if (cpu_id < 0) {
    hwloc_bitmap_free(observed_cpuset);
    hwloc_bitmap_free(target_cpuset);
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::EMPTY_CPUSET;
  }

  if (hwloc_set_cpubind(topology, target_cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) != 0) {
    result->error_number = errno;
    hwloc_bitmap_free(observed_cpuset);
    hwloc_bitmap_free(target_cpuset);
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::BIND_FAILED;
  }
  if (hwloc_get_cpubind(topology, observed_cpuset, HWLOC_CPUBIND_THREAD) != 0 ||
      !hwloc_bitmap_isequal(target_cpuset, observed_cpuset)) {
    result->error_number = errno;
    hwloc_bitmap_free(observed_cpuset);
    hwloc_bitmap_free(target_cpuset);
    hwloc_topology_destroy(topology);
    return TaskQueue::AffinityStatus::VERIFY_FAILED;
  }

  result->active = true;
  result->cpu_id = cpu_id;
  hwloc_bitmap_free(observed_cpuset);
  hwloc_bitmap_free(target_cpuset);
  hwloc_topology_destroy(topology);
  return TaskQueue::AffinityStatus::ACTIVE;
}

}  // namespace

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  try {
    workerThread = std::thread(&TaskQueue::worker, this);
  } catch (...) {
    delete dummy;
    head.store(nullptr, std::memory_order_relaxed);
    tail.store(nullptr, std::memory_order_relaxed);
    throw;
  }
}

TaskQueue::TaskQueue(int first_core_numa_id) : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  affinity_status_ = AffinityStatus::PENDING;
  affinity_numa_id_ = first_core_numa_id;
  try {
    workerThread = std::thread(&TaskQueue::pinned_worker, this, first_core_numa_id);
  } catch (...) {
    delete dummy;
    head.store(nullptr, std::memory_order_relaxed);
    tail.store(nullptr, std::memory_order_relaxed);
    throw;
  }

  {
    std::unique_lock<std::mutex> lock(startup_mtx);
    startup_cv.wait(lock, [&] { return startup_complete; });
  }
  if (affinity_status_ != AffinityStatus::ACTIVE) {
    if (workerThread.joinable()) workerThread.join();
    delete_nodes();
    throw std::runtime_error(std::string("failed to pin TaskQueue worker to NUMA first core: ") + affinity_status() +
                             " (errno=" + std::to_string(affinity_errno_) + ", " + std::strerror(affinity_errno_) +
                             ")");
  }
}

TaskQueue::~TaskQueue() {
  // CPUInfer tasks capture operators backed by WorkerPool.  Finish all work
  // accepted before teardown so the owner can safely destroy that backend
  // after this destructor returns.
  sync(0);
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  delete_nodes();
}

void TaskQueue::delete_nodes() {
  Node* node = head.exchange(nullptr, std::memory_order_relaxed);
  tail.store(nullptr, std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

bool TaskQueue::affinity_requested() const { return affinity_status_ != AffinityStatus::NOT_REQUESTED; }

bool TaskQueue::affinity_active() const { return affinity_status_ == AffinityStatus::ACTIVE; }

int TaskQueue::affinity_numa_id() const { return affinity_numa_id_; }

int TaskQueue::affinity_cpu_id() const { return affinity_cpu_id_; }

long TaskQueue::affinity_native_thread_id() const { return affinity_native_thread_id_; }

const char* TaskQueue::affinity_status() const {
  switch (affinity_status_) {
    case AffinityStatus::NOT_REQUESTED:
      return "not_requested";
    case AffinityStatus::PENDING:
      return "pending";
    case AffinityStatus::ACTIVE:
      return "active";
    case AffinityStatus::TOPOLOGY_INIT_FAILED:
      return "topology_init_failed";
    case AffinityStatus::TOPOLOGY_LOAD_FAILED:
      return "topology_load_failed";
    case AffinityStatus::NUMA_NODE_NOT_FOUND:
      return "numa_node_not_found";
    case AffinityStatus::CORE_NOT_FOUND:
      return "core_not_found";
    case AffinityStatus::CPUSET_ALLOCATION_FAILED:
      return "cpuset_allocation_failed";
    case AffinityStatus::EMPTY_CPUSET:
      return "empty_cpuset";
    case AffinityStatus::BIND_FAILED:
      return "bind_failed";
    case AffinityStatus::VERIFY_FAILED:
      return "verify_failed";
  }
  return "unknown";
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] {
    return pending.load(std::memory_order_acquire) <= allow_n_pending || done.load(std::memory_order_acquire);
  });
}

void TaskQueue::worker() { process_tasks(); }

void TaskQueue::pinned_worker(int numa_id) {
  pthread_setname_np(pthread_self(), "kt_task_queue");
  affinity_native_thread_id_ = static_cast<long>(syscall(SYS_gettid));
  AffinityResult result{false, 0, -1};
  const AffinityStatus status = bind_to_first_core(numa_id, &result);
  {
    std::lock_guard<std::mutex> lock(startup_mtx);
    affinity_status_ = status;
    affinity_errno_ = result.error_number;
    affinity_cpu_id_ = result.cpu_id;
    startup_complete = true;
  }
  startup_cv.notify_one();
  if (!result.active) return;
  process_tasks();
}

void TaskQueue::process_tasks() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      if (next->task) {
        next->task();
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr || done.load(std::memory_order_acquire);
      });
    }
  }
}
