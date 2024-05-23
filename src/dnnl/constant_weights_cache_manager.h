/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

using size_t = std::size_t;

struct ref_count_managed {
  ref_count_managed() = default;
  ref_count_managed(const std::shared_ptr<void> &keep_alive) {
    init(keep_alive);
  }
  void init(const std::shared_ptr<void> &keep_alive) {
    keep_alive_ = keep_alive;
    ref_count_.store(1);
  }

  void ref() { ++ref_count_; }
  void deref() {
    auto newv = --ref_count_;
    if (newv == 0) {
      keep_alive_ = nullptr;
    }
  }

  // atomically check if ref_count_ > 0. if so, ref() the object and return
  // true. Otherwise (if ref_count_==0), return false
  bool check_alive_and_ref() {
    auto oldv = ref_count_.load();
    for (;;) {
      if (oldv <= 0) {
        return false;
      }
      if (ref_count_.compare_exchange_strong(oldv, oldv + 1)) {
        return true;
      }
      // CAS failed, oldv has now the newest known value of ref_count_
    }
  }

  bool is_alive() const { return ref_count_ > 0; }
  void *unsafe_get_ptr() const { return keep_alive_.get(); }

private:
  std::shared_ptr<void> keep_alive_;
  std::atomic<int> ref_count_{0};
};

struct const_cache_proxy : ref_count_managed {
  const_cache_proxy(const std::shared_ptr<void> &keep_alive, void *buffer,
                    size_t size, bool is_lazy)
      : ref_count_managed(keep_alive), size_(size), is_lazy_(is_lazy),
        buffer_(buffer) {}
  ~const_cache_proxy() = default;

  // get the buffer and increment the refcount. If the buffer is evicted,
  // returns null
  void *acquire(int32_t *inited);
  // decrement the refcount
  bool release();

  void *get_buffer_if_not_lazy() const {
    // if (is_lazy_) {
    //     throw std::runtime_error(
    //             "get_buffer_if_not_lazy: The buffer must be lazy");
    // }
    return buffer_;
  }

  size_t size_;
  // if the buffer is lazy-initialized. If false, it should be filled before
  // computation
  bool is_lazy_;

private:
  // raw pointer to the buffer
  void *buffer_;
  // if the buffer has been initialized. calling release() will set this to 1
  int32_t initialized_ = 0;
};

struct dnnl_graph_compiler_context;

// void *(*allocator)(size_t size); // get from dnnl_graph_compiler_context
// void (*deallocator)(void *ptr);  // get from dnnl_graph_compiler_context
void *allocator(size_t size) {
  return std::aligned_alloc(64, size);
}
void deallocator(void *ptr) {
  std::free(ptr);
}

// allocate the const cache buffer and register it to Graph API cache manager
std::shared_ptr<const_cache_proxy>
create_and_register_const_cache(dnnl_graph_compiler_context *ctx, size_t size) {
  // simply allocate buffer and return
  std::shared_ptr<void> base =
      std::shared_ptr<void>{allocator(size), [](void *p) { deallocator(p); }};
  return std::make_shared<const_cache_proxy>(base, base.get(), size, true);
}

// Cached tensor with buffer
struct cached_const_graph_tensor {
  mlir::Value tensor_;
  // mlir::Operation *producer_;
  size_t size_; // original size
  // the base pointer of buf_. buf_ may be cut from a larger buffer buf_base_.
  std::shared_ptr<const_cache_proxy> buf_base_;
  // the offset of buf_ on the base buffer of buf_base_
  size_t offset_ = 0;

  cached_const_graph_tensor(dnnl_graph_compiler_context *ctx, mlir::Value tensor,
                            /*mlir::Operation *producer, */ size_t size) {
    tensor = tensor;
    // producer_ = producer;
    size_ = size;
    buf_base_ = nullptr;
  }
};

size_t divide_and_ceil(size_t x, size_t y) { return (x + y - 1) / y; }

// Manager
struct const_graph_tensor_cache_manager {
  dnnl_graph_compiler_context *ctx;

  // Stores all the cached tensors.
  std::unordered_map<int64_t, std::shared_ptr<cached_const_graph_tensor>>
      value_to_cached_tensor;

  // alloc and set the buf_base_ and offset_ attributes of cache
  void
  alloc(const std::vector<std::shared_ptr<cached_const_graph_tensor>> &caches) {
    size_t total_size = 0;
    for (size_t i = 0; i < caches.size(); i++) {
      auto &v = caches[i];
      if (!v->buf_base_) {
        total_size += divide_and_ceil(v->size_, 64) * 64;
      }
    }
    auto base = create_and_register_const_cache(ctx, total_size);
    llvm::dbgs() << "Alloc base: " << base->get_buffer_if_not_lazy() << ", size: " << total_size << '\n';
    size_t offset = 0;
    for (size_t idx = 0; idx < caches.size(); idx++) {
      auto &v = caches[idx];
      // the update on buf_ is protected by lock_
      if (!v->buf_base_) {
        v->buf_base_ = base;
        v->offset_ = offset;
        llvm::dbgs() << "Alloc base, offset: " << v->offset_ << '\n';
        offset += divide_and_ceil(v->size_, 64) * 64;
      }
    }
  }

  // Get a hash value from a tensor.
  int64_t hash_tensor(mlir::Value tensor) { return 1; }

  std::shared_ptr<cached_const_graph_tensor> add_tensor(mlir::Value tensor,
                                                        size_t buf_size) {
    auto ret = std::make_shared<cached_const_graph_tensor>(ctx, tensor, buf_size);
    int64_t key = hash_tensor(tensor);
    value_to_cached_tensor[key] = ret;
    return ret;
  }

  void remove(mlir::Value tensor) {
    int64_t key = hash_tensor(tensor);
    if (value_to_cached_tensor.find(key) != value_to_cached_tensor.end()) {
      value_to_cached_tensor.erase(key);
    }
  }

  // singleton
  static std::shared_ptr<const_graph_tensor_cache_manager> get_cache() {
    static std::shared_ptr<const_graph_tensor_cache_manager> c =
        std::make_shared<const_graph_tensor_cache_manager>();
    return c;
  }
};
