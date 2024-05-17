//===- TileUsingInterface.h - Tiling ops using TilingInterface --*- C++ -*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#ifndef TEMPORARY_TILEUSINGINTERFACE_X_H
#define TEMPORARY_TILEUSINGINTERFACE_X_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace mlir {
namespace scfX {
// Extension for upstream `tileAndFuseProducerOfSlice`
std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                           tensor::ExtractSliceOp candidateSliceOp,
                           MutableArrayRef<LoopLikeOpInterface> loops);

// Extension for upcoming upstream `tileAndFuseConsumerOfSlice`
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerOfSlice(RewriterBase &rewriter, Operation *candidateSliceOp);

} // namespace scfX
} // namespace mlir

#endif
