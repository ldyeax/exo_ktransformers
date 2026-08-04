/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class TaskQueue {
 public:
  enum class AffinityStatus {
    NOT_REQUESTED,
    PENDING,
    ACTIVE,
    TOPOLOGY_INIT_FAILED,
    TOPOLOGY_LOAD_FAILED,
    NUMA_NODE_NOT_FOUND,
    CORE_NOT_FOUND,
    CPUSET_ALLOCATION_FAILED,
    EMPTY_CPUSET,
    BIND_FAILED,
    VERIFY_FAILED,
  };

  TaskQueue();
  explicit TaskQueue(int first_core_numa_id);
  ~TaskQueue();

  void enqueue(std::function<void()>);

  void sync(size_t allow_n_pending);

  bool affinity_requested() const;
  bool affinity_active() const;
  int affinity_numa_id() const;
  int affinity_cpu_id() const;
  long affinity_native_thread_id() const;
  const char* affinity_status() const;

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()>& t) : task(t), next(nullptr) {}
  };

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  std::condition_variable cv;

  std::mutex startup_mtx;
  std::condition_variable startup_cv;
  bool startup_complete = false;
  AffinityStatus affinity_status_ = AffinityStatus::NOT_REQUESTED;
  int affinity_errno_ = 0;
  int affinity_numa_id_ = -1;
  int affinity_cpu_id_ = -1;
  long affinity_native_thread_id_ = -1;

  void worker();
  void pinned_worker(int);
  void process_tasks();
  void delete_nodes();
};

#endif
