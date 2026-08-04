#include <hwloc.h>
#include <sched.h>

#include <atomic>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cpu_backend/task_queue.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int first_numa_os_index() {
  hwloc_topology_t topology = nullptr;
  require(hwloc_topology_init(&topology) == 0, "hwloc_topology_init failed");
  if (hwloc_topology_load(topology) != 0) {
    hwloc_topology_destroy(topology);
    throw std::runtime_error("hwloc_topology_load failed");
  }
  hwloc_obj_t numa_object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, nullptr);
  if (numa_object == nullptr) {
    hwloc_topology_destroy(topology);
    throw std::runtime_error("machine has no NUMA node");
  }
  const int numa_id = static_cast<int>(numa_object->os_index);
  hwloc_topology_destroy(topology);
  return numa_id;
}

void test_pinned_worker() {
  TaskQueue queue(first_numa_os_index());
  require(queue.affinity_requested(), "affinity was not requested");
  require(queue.affinity_active(), "affinity was not active");
  require(std::string(queue.affinity_status()) == "active", "affinity status was not active");
  require(queue.affinity_cpu_id() >= 0, "affinity CPU ID was not reported");
  require(queue.affinity_native_thread_id() > 0, "native thread ID was not reported");

  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  require(sched_getaffinity(static_cast<pid_t>(queue.affinity_native_thread_id()), sizeof(affinity), &affinity) == 0,
          "sched_getaffinity failed for TaskQueue thread");
  require(CPU_COUNT(&affinity) == 1, "TaskQueue thread affinity contains more than one CPU");
  require(CPU_ISSET(queue.affinity_cpu_id(), &affinity), "TaskQueue thread is not bound to its reported CPU");

  std::atomic<int> executed_cpu{-1};
  queue.enqueue([&] { executed_cpu.store(sched_getcpu(), std::memory_order_release); });
  queue.sync(0);
  require(executed_cpu.load(std::memory_order_acquire) == queue.affinity_cpu_id(),
          "queued work did not execute on the reported CPU");
}

void test_constructor_failure_is_joined() {
  bool rejected = false;
  try {
    TaskQueue invalid_queue(INT_MAX);
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("numa_node_not_found") != std::string::npos;
  }
  require(rejected, "invalid NUMA affinity did not fail closed");

  TaskQueue healthy_queue;
  std::atomic<bool> executed{false};
  healthy_queue.enqueue([&] { executed.store(true, std::memory_order_release); });
  healthy_queue.sync(0);
  require(executed.load(std::memory_order_acquire), "TaskQueue was unusable after constructor rejection");
}

}  // namespace

int main() {
  try {
    test_pinned_worker();
    test_constructor_failure_is_joined();
  } catch (const std::exception& error) {
    std::cerr << "test_task_queue_affinity: " << error.what() << '\n';
    return 1;
  }
  std::cout << "test_task_queue_affinity: passed\n";
  return 0;
}
