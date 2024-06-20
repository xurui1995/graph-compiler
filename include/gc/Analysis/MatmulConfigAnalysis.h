//===- MatmulConfigAnalysis.h - Graph Compiler analysis pass ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H
#define MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <llvm/Support/Debug.h>
#include <memory>
#include <numeric>

namespace mlir {
namespace gc {

using namespace mlir;

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads();
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel);
};

struct MatmulConfig {
  uint32_t MBlock, NBlock, KBlock;
  uint32_t MThreads, NThreads, KThreads;
  uint32_t innerMostMBlock, innerMostNBlock, innerMostKBlock;
};

enum DimType { Batch, M, N, K };

[[maybe_unused]] static SmallVector<unsigned>
extractDimTypeIdx(ArrayRef<DimType> tyList, DimType ty) {
  SmallVector<unsigned> idxList;
  for (auto [idx, type] : llvm::enumerate(tyList)) {
    if (type == ty) {
      idxList.push_back(idx);
    }
  }
  return idxList;
}

static FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  if (isa<linalg::MatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm2DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm4DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::BatchMatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::Batch, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::N}};
  }
  return failure();
}

struct MatmulConfigAnalysis {
public:
  explicit MatmulConfigAnalysis(Operation *root);
  MatmulConfig getConfig() { return config; }

private:
  MatmulConfig config;
};

} // namespace gc
} // namespace mlir

#endif