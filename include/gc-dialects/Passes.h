//===- Passes.h - Graph Compiler passes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class OpBuilder;
class SymbolTable;
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func

namespace gc {

std::unique_ptr<Pass> createMergeAllocPass();

#define GEN_PASS_DECL
#include "gc-dialects/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc-dialects/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
