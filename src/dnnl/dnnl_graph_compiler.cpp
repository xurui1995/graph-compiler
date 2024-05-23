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

#include "dnnl_graph_compiler.h"
#include "gc_version.h"
#include <memory>
#include <new>
#include <vector>
#include <string_view>

#include "constant_weights_cache_manager.h"

#if defined _WIN32 || defined __CYGWIN__
#define GC_DLL_EXPORT __declspec(dllexport)
#else
#define GC_DLL_EXPORT __attribute__((visibility("default")))
#endif

// dnnl_graph_compiler.h interface implementation.
// TODO: Implement.

typedef void (*FuncType)(...);

struct dnnl_graph_compiler_executable {
  // TODO: Implement

  std::vector<int64_t> folded_args_index;
  std::size_t num_inputs;
  std::size_t num_outputs;

  std::vector<void *> folded_args;
  bool is_inited = false;

  FuncType fold;
  FuncType compute;

  dnnl_graph_compiler_executable() {
    // set folded_args_index, num_inputs and num_outputs
    // set folded_args
    auto manager = const_graph_tensor_cache_manager::get_cache();
    for (size_t idx = 0; idx < folded_args_index.size(); ++idx) {
      int64_t key = folded_args_index[idx];
      void *buffer_base = manager->value_to_cached_tensor[key]->buf_base_->acquire();
      size_t offset = manager->value_to_cached_tensor[key]->offset_;
      folded_args.push_back(buffer_base + offset);
    }
    // set fold and compute
  }

  void execute(dnnl_graph_compiler_tensor *inputs,
               dnnl_graph_compiler_tensor *outputs) const;
};

struct dnnl_graph_compiler {
  const dnnl_graph_compiler_context ctx;

  explicit dnnl_graph_compiler(const dnnl_graph_compiler_context *context)
      // TODO: Initialize ctx with context or defaults if context is nullptr
      : ctx() {}

  [[nodiscard]] std::unique_ptr<const dnnl_graph_compiler_executable>
  compile(const std::string_view &graph_json) const;
};

GC_DLL_EXPORT const dnnl_graph_compiler_version *
dnnl_graph_compiler_get_version(void) {
  static const dnnl_graph_compiler_version ver = {
      .api_version = {DNNL_GC_API_V_MAJOR, DNNL_GC_API_V_MINOR,
                      DNNL_GC_API_V_PATCH,
                      DNNL_GC_API_V_HASH}, // version defined by oneDNN
      .gc_version = {
          GC_VERSION_MAJOR, GC_VERSION_MINOR, GC_VERSION_PATCH,
          GC_VERSION_HASH}}; // version defined by graph compiler itself
  return &ver;
}

GC_DLL_EXPORT dnnl_status_t
dnnl_graph_compiler_create(const struct dnnl_graph_compiler_context *ctx,
                           const struct dnnl_graph_compiler **gc) {
  try {
    *gc = new dnnl_graph_compiler(ctx);
    (*gc)->const_weights_cache_manager = new const_graph_tensor_cache_manager();
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

GC_DLL_EXPORT void
dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc) {
  delete gc;
}

GC_DLL_EXPORT dnnl_status_t dnnl_graph_compiler_compile(
    const dnnl_graph_compiler *gc, const char *graph_json,
    const struct dnnl_graph_compiler_executable **exe) {
  try {
    auto ptr = gc->compile(std::string_view(graph_json));
    *exe = ptr.release();
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

GC_DLL_EXPORT void dnnl_graph_compiler_destroy_executable(
    const struct dnnl_graph_compiler *gc,
    const struct dnnl_graph_compiler_executable *exe) {
  delete exe;
}

GC_DLL_EXPORT dnnl_status_t dnnl_graph_compiler_execute(
    const struct dnnl_graph_compiler *gc,
    const struct dnnl_graph_compiler_executable *exe,
    dnnl_graph_compiler_tensor *inputs, dnnl_graph_compiler_tensor *outputs) {
  try {
    exe->execute(inputs, outputs);
    return dnnl_success;
  } catch (const std::bad_alloc &e) {
    return dnnl_out_of_memory;
  } catch (...) {
    // TODO: Add error handling
    return dnnl_runtime_error;
  }
}

std::unique_ptr<const dnnl_graph_compiler_executable>
dnnl_graph_compiler::compile(const std::string_view &graph_json) const {
  // TODO: Implement
  auto exe = std::unique_ptr<const dnnl_graph_compiler_executable>(
      new dnnl_graph_compiler_executable());

  /*
  // Call constant_weights_folding_pass to set attributes of exe and allocate buffers.
  exe->num_constant_inputs = ...;
  exe->num_inputs = ...;
  exe->num_outputs = ...;

  // Lowered functions.
  exe->fold = ...;
  exe->compute = ...;
  */
  return exe;
}

void dnnl_graph_compiler_executable::execute(
    dnnl_graph_compiler_tensor *inputs,
    dnnl_graph_compiler_tensor *outputs) const {
  // TODO: Implement

  void *const_input_tensors = inputs->data;

  // whether the constant input tensors have been folded.
  if (!is_inited) {
    // construct argument list for fold
    call_fold(const_input_tensors, folded_args);
    is_inited = true;
  }

  // construct argument list by replacing constant inputs with folded inputs; call compute
  call_compute(inputs, folded_args, outputs);
}
