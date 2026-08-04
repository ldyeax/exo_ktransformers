#include <hwloc.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cpu_backend/task_queue.h"
#include "cpu_backend/worker_pool.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

long native_thread_id() { return static_cast<long>(syscall(SYS_gettid)); }

struct NumaTopology {
  int numa_id;
  int core_count;
  int processing_unit_count;
};

NumaTopology test_numa_topology() {
  hwloc_topology_t topology = nullptr;
  require(hwloc_topology_init(&topology) == 0, "hwloc_topology_init failed");
  if (hwloc_topology_load(topology) != 0) {
    hwloc_topology_destroy(topology);
    throw std::runtime_error("hwloc_topology_load failed");
  }
  hwloc_obj_t numa_object = nullptr;
  while ((numa_object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_object)) != nullptr) {
    const int core_count = hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE);
    const int processing_unit_count =
        hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_PU);
    if (core_count >= 2 && processing_unit_count >= core_count) {
      const NumaTopology result{static_cast<int>(numa_object->os_index), core_count, processing_unit_count};
      hwloc_topology_destroy(topology);
      return result;
    }
  }
  hwloc_topology_destroy(topology);
  throw std::runtime_error("no usable NUMA node found");
}

int expected_cpu_id(const NumaTopology& selected, int worker_index) {
  hwloc_topology_t topology = nullptr;
  require(hwloc_topology_init(&topology) == 0, "hwloc_topology_init failed");
  require(hwloc_topology_load(topology) == 0, "hwloc_topology_load failed");
  hwloc_obj_t numa_object = nullptr;
  while ((numa_object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_object)) != nullptr &&
         static_cast<int>(numa_object->os_index) != selected.numa_id) {
  }
  require(numa_object != nullptr, "selected NUMA node disappeared");

  hwloc_obj_t processing_unit = nullptr;
  if (worker_index < selected.core_count) {
    hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE, worker_index);
    processing_unit = hwloc_get_obj_inside_cpuset_by_type(topology, core->cpuset, HWLOC_OBJ_PU, 0);
  } else {
    int remaining = worker_index - selected.core_count;
    for (int sibling_index = 1; processing_unit == nullptr; ++sibling_index) {
      bool found_any = false;
      for (int core_index = 0; core_index < selected.core_count; ++core_index) {
        hwloc_obj_t core =
            hwloc_get_obj_inside_cpuset_by_type(topology, numa_object->cpuset, HWLOC_OBJ_CORE, core_index);
        hwloc_obj_t candidate =
            hwloc_get_obj_inside_cpuset_by_type(topology, core->cpuset, HWLOC_OBJ_PU, sibling_index);
        if (candidate == nullptr) continue;
        found_any = true;
        if (remaining-- == 0) {
          processing_unit = candidate;
          break;
        }
      }
      require(found_any, "worker index exceeds NUMA-local processing units");
    }
  }
  require(processing_unit != nullptr, "expected processing unit was not found");
  const int cpu_id = static_cast<int>(processing_unit->os_index);
  hwloc_topology_destroy(topology);
  return cpu_id;
}

void require_singleton_affinity(long thread_id, int cpu_id) {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  require(sched_getaffinity(static_cast<pid_t>(thread_id), sizeof(affinity), &affinity) == 0,
          "sched_getaffinity failed");
  require(CPU_COUNT(&affinity) == 1, "worker affinity is not a singleton");
  require(CPU_ISSET(cpu_id, &affinity), "worker affinity differs from telemetry");
}

WorkerPoolConfig one_subpool(int numa_id, int worker_count) { return WorkerPoolConfig{1, {numa_id}, {worker_count}}; }

void test_default_path_is_unchanged(const NumaTopology& topology) {
  unsetenv("KT_SINGLE_NUMA_INLINE_DISPATCH");
  WorkerPool pool(one_subpool(topology.numa_id, 2));
  require(!pool.single_numa_inline_dispatch_environment_enabled(), "inline environment unexpectedly enabled");
  require(pool.single_numa_inline_dispatch_eligible(), "one subpool was not eligible");
  require(!pool.single_numa_inline_dispatch_requested(), "inline dispatch unexpectedly requested");
  require(!pool.single_numa_inline_dispatch_active(), "inline dispatch unexpectedly active");
  require(std::string(pool.single_numa_inline_dispatch_status()) == "not_requested", "default inline status changed");
  require(pool.single_numa_inline_dispatch_worker_count() == 1, "default distributor worker was elided");

  const long caller_thread_id = native_thread_id();
  std::atomic<long> callback_thread_id{-1};
  pool.dispense_backend()->do_numa_job([&](int subpool_index) {
    callback_thread_id.store(native_thread_id() + subpool_index, std::memory_order_release);
  });
  require(callback_thread_id.load(std::memory_order_acquire) != caller_thread_id,
          "default distributor unexpectedly executed inline");
}

void test_ineligible_request_fails_closed(const NumaTopology& topology) {
  setenv("KT_SINGLE_NUMA_INLINE_DISPATCH", "1", 1);
  bool rejected = false;
  try {
    WorkerPool invalid(WorkerPoolConfig{2, {topology.numa_id, topology.numa_id}, {1, 1}});
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("exactly one physical NUMA subpool") != std::string::npos;
  }
  require(rejected, "ineligible inline request did not fail closed");

  WorkerPool recovered(one_subpool(topology.numa_id, 2));
  require(recovered.single_numa_inline_dispatch_active(), "eligible construction failed after rejection");
}

void test_inline_worker_zero_and_oversubscribed_affinity(const NumaTopology& topology) {
  setenv("KT_SINGLE_NUMA_INLINE_DISPATCH", "1", 1);
  const int requested_worker_count = std::min(72, topology.processing_unit_count);
  require(requested_worker_count >= topology.core_count, "test NUMA node cannot exercise physical-core mapping");

  WorkerPool pool(one_subpool(topology.numa_id, requested_worker_count));
  TaskQueue queue(topology.numa_id);
  require(pool.single_numa_inline_dispatch_active(), "inline dispatch was not active");
  require(pool.single_numa_inline_dispatch_worker_count() == 0, "inline mode retained a distributor worker");
  require(pool.get_thread_num() == requested_worker_count, "configured worker count telemetry is wrong");

  std::vector<long> observed_thread_ids(requested_worker_count, -1);
  std::vector<int> observed_cpu_ids(requested_worker_count, -1);
  std::atomic<int> callback_subpool_index{-1};
  std::atomic<long> callback_thread_id{-1};
  std::atomic<int> callback_cpu_id{-1};
  std::atomic<int> computed_tasks{0};
  queue.enqueue([&] {
    pool.dispense_backend()->do_numa_job([&](int subpool_index) {
      callback_subpool_index.store(subpool_index, std::memory_order_release);
      callback_thread_id.store(native_thread_id(), std::memory_order_release);
      callback_cpu_id.store(sched_getcpu(), std::memory_order_release);
      pool.get_subpool(subpool_index)
          ->do_work_stealing_job(
              requested_worker_count,
              [&](int logical_worker_id) {
                observed_thread_ids[logical_worker_id] = native_thread_id();
                observed_cpu_ids[logical_worker_id] = sched_getcpu();
              },
              [&](int) { computed_tasks.fetch_add(1, std::memory_order_relaxed); }, nullptr);
    });
  });
  queue.sync(0);

  require(callback_subpool_index.load(std::memory_order_acquire) == 0,
          "inline callback received a physical rather than logical subpool index");
  require(callback_thread_id.load(std::memory_order_acquire) == queue.affinity_native_thread_id(),
          "inline callback did not execute on TaskQueue");
  require(callback_cpu_id.load(std::memory_order_acquire) == queue.affinity_cpu_id(),
          "inline callback did not execute on TaskQueue CPU");
  require(computed_tasks.load(std::memory_order_acquire) == requested_worker_count, "worker pool lost tasks");
  require(pool.single_numa_inline_dispatch_count() == 1, "inline dispatch count is wrong");
  require(pool.single_numa_inline_dispatch_last_native_thread_id() == queue.affinity_native_thread_id(),
          "inline native-thread telemetry is wrong");
  require(pool.single_numa_inline_dispatch_last_cpu_id() == queue.affinity_cpu_id(), "inline CPU telemetry is wrong");
  require(pool.single_numa_inline_dispatch_last_worker_pool_thread_id() == 0, "TaskQueue was not logical worker 0");
  require(pool.subpool_last_caller_native_thread_id(0) == queue.affinity_native_thread_id(),
          "InNumaPool caller telemetry differs from TaskQueue");

  std::vector<int> affinity_cpu_ids = pool.subpool_worker_affinity_cpu_ids(0);
  std::vector<long> native_thread_ids = pool.subpool_worker_native_thread_ids(0);
  std::vector<std::string> affinity_statuses = pool.subpool_worker_affinity_statuses(0);
  affinity_cpu_ids[0] = queue.affinity_cpu_id();
  native_thread_ids[0] = queue.affinity_native_thread_id();
  affinity_statuses[0] = "active";
  std::vector<int> unique_cpu_ids;
  for (int worker_index = 0; worker_index < requested_worker_count; ++worker_index) {
    require(affinity_statuses[worker_index] == "active", "worker affinity was not active");
    require(affinity_cpu_ids[worker_index] == expected_cpu_id(topology, worker_index),
            "worker CPU differs from deterministic mapping");
    require(native_thread_ids[worker_index] == observed_thread_ids[worker_index],
            "worker native TID differs from executed role");
    require(observed_cpu_ids[worker_index] == affinity_cpu_ids[worker_index],
            "worker executed outside its reported CPU");
    require_singleton_affinity(native_thread_ids[worker_index], affinity_cpu_ids[worker_index]);
    unique_cpu_ids.push_back(affinity_cpu_ids[worker_index]);
  }
  std::sort(unique_cpu_ids.begin(), unique_cpu_ids.end());
  require(std::adjacent_find(unique_cpu_ids.begin(), unique_cpu_ids.end()) == unique_cpu_ids.end(),
          "two logical workers share one processing unit");
  if (requested_worker_count > topology.core_count) {
    require(affinity_cpu_ids[topology.core_count] != affinity_cpu_ids[0],
            "first oversubscribed worker wrapped onto worker0 primary PU");
  }

  std::atomic<int> repeated_callbacks{0};
  std::atomic<bool> caught_expected_exception{false};
  queue.enqueue([&] {
    for (int iteration = 0; iteration < 256; ++iteration) {
      pool.dispense_backend()->do_numa_job(
          [&](int subpool_index) { repeated_callbacks.fetch_add(subpool_index + 1, std::memory_order_relaxed); });
    }
    try {
      pool.dispense_backend()->do_numa_job([](int) { throw std::runtime_error("expected inline failure"); });
    } catch (const std::runtime_error&) {
      caught_expected_exception.store(true, std::memory_order_release);
    }
    pool.dispense_backend()->do_numa_job(
        [&](int subpool_index) { repeated_callbacks.fetch_add(subpool_index + 1, std::memory_order_relaxed); });
  });
  queue.sync(0);
  require(repeated_callbacks.load(std::memory_order_acquire) == 257, "inline dispatcher did not survive reuse");
  require(caught_expected_exception.load(std::memory_order_acquire), "inline exception was not recoverable");
  require(pool.single_numa_inline_dispatch_count() == 259, "reuse dispatch count is wrong");
  require(pool.single_numa_inline_exception_count() == 1, "inline exception count is wrong");
}

void test_pending_teardown_drains_before_backend_release(const NumaTopology& topology) {
  setenv("KT_SINGLE_NUMA_INLINE_DISPATCH", "1", 1);
  WorkerPool pool(one_subpool(topology.numa_id, 2));
  auto queue = std::make_unique<TaskQueue>(topology.numa_id);
  std::atomic<bool> first_task_started{false};
  std::atomic<bool> release_first_task{false};
  std::atomic<int> completed_tasks{0};
  std::atomic<bool> destructor_returned{false};

  queue->enqueue([&] {
    pool.dispense_backend()->do_numa_job([&](int subpool_index) {
      first_task_started.store(true, std::memory_order_release);
      while (!release_first_task.load(std::memory_order_acquire)) std::this_thread::yield();
      pool.get_subpool(subpool_index)->do_work_stealing_job(2, [&](int) {});
      completed_tasks.fetch_add(1, std::memory_order_release);
    });
  });
  queue->enqueue([&] {
    pool.dispense_backend()->do_numa_job([&](int) { completed_tasks.fetch_add(1, std::memory_order_release); });
  });
  while (!first_task_started.load(std::memory_order_acquire)) std::this_thread::yield();
  std::thread teardown_thread([&] {
    queue.reset();
    destructor_returned.store(true, std::memory_order_release);
  });
  require(!destructor_returned.load(std::memory_order_acquire), "TaskQueue destructor did not wait for pending work");
  release_first_task.store(true, std::memory_order_release);
  teardown_thread.join();
  require(destructor_returned.load(std::memory_order_acquire), "TaskQueue destructor did not return");
  require(completed_tasks.load(std::memory_order_acquire) == 2, "TaskQueue destructor abandoned accepted work");
  require(pool.single_numa_inline_dispatch_count() == 2, "teardown did not cleanly drain inline dispatches");
}

}  // namespace

int main() {
  try {
    const NumaTopology topology = test_numa_topology();
    test_default_path_is_unchanged(topology);
    test_ineligible_request_fails_closed(topology);
    test_inline_worker_zero_and_oversubscribed_affinity(topology);
    test_pending_teardown_drains_before_backend_release(topology);
    unsetenv("KT_SINGLE_NUMA_INLINE_DISPATCH");
  } catch (const std::exception& error) {
    std::cerr << "test_single_numa_inline_dispatch: " << error.what() << '\n';
    return 1;
  }
  std::cout << "test_single_numa_inline_dispatch: passed\n";
  return 0;
}
