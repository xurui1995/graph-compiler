//===-- TargetDescriptionAnalysis.h - target description class --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_TARGETDESCRIPTIONANALYSIS_H
#define MLIR_ANALYSIS_TARGETDESCRIPTIONANALYSIS_H

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace gc {

using namespace mlir;

class TargetDescriptionAnalysisBase {
public:
  TargetDescriptionAnalysisBase(Operation *op, std::string device)
      : ctx(op->getContext()), device(device),
        layout(isa<ModuleOp>(op) ? dyn_cast<ModuleOp>(op)
                                 : op->getParentOfType<ModuleOp>()),
        loc(op->getLoc()) {}
  // get the device ID
  std::string getDevice() { return device; }

  // get the MLIR context
  MLIRContext *getContext() { return ctx; }

  // get the data layout
  DataLayout getLayout() { return layout; }

  // get the property value by key
  std::optional<Attribute> getPropertyValue(StringRef key);

  // get the location
  Location getLocation() { return loc; }

  // check if the property exists
  bool hasProperty(StringRef key) { return getPropertyValue(key).has_value(); }

private:
  MLIRContext *ctx;
  std::string device;
  DataLayout layout;
  Location loc;
};

class CPUTargetDescriptionAnalysis : public TargetDescriptionAnalysisBase {
public:
  static constexpr StringLiteral kL1CacheSize = "L1_cache_size_in_bytes";
  static constexpr StringLiteral kL2CacheSize = "L2_cache_size_in_bytes";
  static constexpr StringLiteral kL3CacheSize = "L3_cache_size_in_bytes";
  static constexpr StringLiteral kMaxVectorWidth = "max_vector_width";
  static constexpr StringLiteral kNumThreads = "num_threads";

  // get runtime OMP_NUM_THREADS
  size_t getNumThreads();

  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel);

  // get the maximum vector length in bits
  size_t getMaxVectorWidth();

  CPUTargetDescriptionAnalysis(Operation *op)
      : TargetDescriptionAnalysisBase(op, "CPU") {}
};

} // namespace gc
} // namespace mlir

#endif