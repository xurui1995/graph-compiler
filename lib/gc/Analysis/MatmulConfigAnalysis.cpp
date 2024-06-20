//===- MatmulConfigAnalysis.cpp - Propagate packing on linalg named ops *-
// C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "gc/Analysis/MatmulConfigAnalysis.h"

namespace mlir {
namespace gc {

#define DEBUG_TYPE "matmul-config-analysis"

MatmulConfigAnalysis::MatmulConfigAnalysis(Operation *root) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(root)) {
    // TODO: build a more complex heuristic to determine the best tiling
    auto oprandDimType = *getOprandDimType(linalgOp);
    // get the origin M,N,K size
    auto MDimTypeIdx = extractDimTypeIdx(oprandDimType[0], DimType::M);
    auto KDimTypeIdx = extractDimTypeIdx(oprandDimType[0], DimType::K);
    auto NDimTypeIdx = extractDimTypeIdx(oprandDimType[1], DimType::N);
    auto M = 1, N = 1, K = 1;
    for (auto [s, dimType] :
         llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(0)),
                   oprandDimType[0])) {
      if (dimType == DimType::M) {
        M *= s;
      } else if (dimType == DimType::K) {
        K *= s;
      }
    }
    for (auto [s, dimType] :
         llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(1)),
                   oprandDimType[0])) {
      if (dimType == DimType::N) {
        N *= s;
      }
    }

    // innermost Block, if the layout is blockied layout, the innermost block
    // will derived from the layout directly
    auto defaultBlock = 32;
    config.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
    config.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
    config.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;
    if (MDimTypeIdx.size() > 1) {
      config.innerMostMBlock = 1;
      for (auto i = 1UL; i < MDimTypeIdx.size(); i++) {
        config.innerMostMBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(0))[MDimTypeIdx[i]];
      }
    }
    if (KDimTypeIdx.size() > 1) {
      config.innerMostKBlock = 1;
      for (auto i = 1UL; i < KDimTypeIdx.size(); i++) {
        config.innerMostKBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(0))[KDimTypeIdx[i]];
      }
    }
    if (NDimTypeIdx.size() > 1) {
      config.innerMostNBlock = 1;
      for (auto i = 1UL; i < NDimTypeIdx.size(); i++) {
        config.innerMostNBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(1))[NDimTypeIdx[i]];
      }
    }

    // Number of block
    auto MNumBlock = M / config.innerMostMBlock;
    auto NNumBlock = N / config.innerMostNBlock;
    auto KNumBlock = K / config.innerMostKBlock;

    // Threads
    config.MThreads = 32;
    config.NThreads = 1;
    config.KThreads = 1;

    // Block
    config.MBlock = (int)llvm::divideCeil(MNumBlock, config.MThreads) *
                    config.innerMostMBlock;
    config.NBlock = (int)llvm::divideCeil(NNumBlock, config.NThreads) *
                    config.innerMostNBlock;
    config.KBlock = (int)llvm::divideCeil(KNumBlock, config.KThreads) *
                    config.innerMostKBlock;
    config.MBlock = 128;
    config.NBlock = 128;
    config.KBlock = 128;
    config.MThreads = 2;
    config.NThreads = 2;
    config.KThreads = 1;
  }
}
} // namespace gc
} // namespace mlir