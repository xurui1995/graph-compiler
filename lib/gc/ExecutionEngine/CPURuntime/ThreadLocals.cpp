//===-- ThreadLocals.cpp - threadlocal ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/MemoryPool/ThreadLocals.h"
#include <functional>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

namespace mlir {
namespace gc {

// if registry is destoryed, it will be set to true
static bool registry_destroyed = false;

thread_local_registry_t::thread_local_registry_t() = default;

// release all registered TLS resources. The registry still keeps track of
// the TLS objects
void thread_local_registry_t::release() {
  std::lock_guard<std::mutex> guard(lock_);
  dead_threads_.clear();
  for (auto node : tls_buffers_) {
    node->main_memory_pool_.release();
    node->thread_memory_pool_.release();
    node->additional_->dyn_threadpool_mem_pool_.release();
  }
}

thread_local_registry_t::~thread_local_registry_t() { release(); }

void thread_local_registry_t::for_each_tls_additional(
    const std::function<void(thread_local_buffer_t::additional_t *)> &f) {
  for (auto &v : tls_buffers_) {
    f(v->additional_.get());
  }
  for (auto &v : dead_threads_) {
    f(v.get());
  }
}

struct registry_guard_t {
  std::shared_ptr<thread_local_registry_t> ptr_ =
      std::make_shared<thread_local_registry_t>();
  ~registry_guard_t() {
    ptr_->release();
    registry_destroyed = true;
  }
};

const std::shared_ptr<thread_local_registry_t> &get_thread_locals_registry() {
  static registry_guard_t registry;
  return registry.ptr_;
}

thread_local_buffer_t::additional_t::additional_t() {
  assert(!registry_destroyed);
  registry_ = get_thread_locals_registry();
}

// register itself into registry
thread_local_buffer_t::thread_local_buffer_t()
    : additional_(std::unique_ptr<additional_t>(new additional_t{})) {
  auto &registry = *additional_->registry_;
  std::lock_guard<std::mutex> guard(registry.lock_);
  registry.tls_buffers_.emplace_back(this);
  cur_pos_ = registry.tls_buffers_.end();
  // cur_pos should point to the current buffer iterator in tls_buffers_
  --cur_pos_;
}

// the destructor of TLS. It will unregister `this` pointer in registry. Note
// that C++ standard makes sure that thread local objects are destroyed
// before "static lifetime" objects. However, OpenMP seems not have clearly
// specified whether/when C++11 thread_local is destructed. Experiences on
// g++ 8.4.0 shows that the destructor of thread_local objects in OpenMP threads
// are NEVER called! So we still need to check if registry has already been
// destructed
thread_local_buffer_t::~thread_local_buffer_t() {
  // C++ compiler will call ~thread_local_buffer_t() first and then call dtor
  // of its fields. Note that after ~thread_local_buffer_t() returns, the
  // lock will be released and dtors of member fields will not be protected by
  // the lock. This is OK because we have removed the reference to `this`
  // pointer from the registry and the registry has no chance to call
  // release() on `this` any more. So there will be only one thread calling
  // dtor/release() on the members at the same time

  // move out the ownership of the registry_ to avoid self-referencing
  auto p_registry = std::move(additional_->registry_);
  std::lock_guard<std::mutex> guard(p_registry->lock_);
  assert(*cur_pos_ = this);
  // remove from the tls_buffers_ in registry
  p_registry->tls_buffers_.erase(cur_pos_);
  // we have already moved the registry_ inside additional_)
  p_registry->dead_threads_.emplace_back(std::move(additional_));
}

void release_runtime_memory() {
  // in case registry_guard already destroyed
  if (registry_destroyed) {
    return;
  }
  get_thread_locals_registry()->release();
}

} // namespace gc
} // namespace mlir