//===-- TargetDescriptionAnalysis.cpp - target description impl -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include <limits>
#include <llvm/Support/Debug.h>

namespace mlir {
namespace gc {

#define DEBUG_TYPE "target-description-analysis"

// default values for properties
static llvm::DenseMap<StringRef, int64_t> CPUTargetDeafultValueMap = {
    {CPUTargetDescriptionAnalysis::kNumThreads, 1},
    {CPUTargetDescriptionAnalysis::kL1CacheSize, 32 * 1024},
    {CPUTargetDescriptionAnalysis::kL2CacheSize, 32 * 32 * 1024},
    {CPUTargetDescriptionAnalysis::kL3CacheSize, 32 * 32 * 1024},
    {CPUTargetDescriptionAnalysis::kMaxVectorWidth, 512},
};

static void emitNotFoundWarning(Location loc, StringRef key) {
  mlir::emitWarning(loc) << key << " not found, using default value "
                         << CPUTargetDeafultValueMap[key];
}

static int64_t getIntFromAttribute(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isSignedInteger())
      return intAttr.getSInt();
    else if (intAttr.getType().isUnsignedInteger())
      return intAttr.getUInt();
    else
      return intAttr.getInt();
  }
  llvm_unreachable("Not an integer attribute");
}

std::optional<Attribute>
TargetDescriptionAnalysisBase::getPropertyValue(StringRef key) {
  return layout.getDevicePropertyValue(
      Builder(getContext()).getStringAttr(getDevice() /* device ID*/),
      Builder(getContext()).getStringAttr(key));
}

size_t CPUTargetDescriptionAnalysis::getNumThreads() {
  std::optional<Attribute> numThreads = getPropertyValue(kNumThreads);

  if (numThreads && isa<IntegerAttr>(*numThreads))
    return getIntFromAttribute(*numThreads);
  emitNotFoundWarning(getLocation(), kNumThreads);
  return CPUTargetDeafultValueMap[kNumThreads];
}

size_t CPUTargetDescriptionAnalysis::getCacheSize(uint8_t cacheLevel) {
  assert(cacheLevel > 0 && cacheLevel < 4 && "Invalid cache level");
  StringLiteral key = "";
  if (cacheLevel == 1)
    key = kL1CacheSize;
  else if (cacheLevel == 2)
    key = kL2CacheSize;
  else if (cacheLevel == 3)
    key = kL3CacheSize;

  std::optional<Attribute> cacheSize = getPropertyValue(key);
  if (cacheSize && isa<IntegerAttr>(*cacheSize))
    return getIntFromAttribute(*cacheSize);

  emitNotFoundWarning(getLocation(), key);
  return CPUTargetDeafultValueMap[key];
}

size_t CPUTargetDescriptionAnalysis::getMaxVectorWidth() {
  std::optional<Attribute> maxVectorWidth = getPropertyValue(kMaxVectorWidth);
  if (maxVectorWidth && isa<IntegerAttr>(*maxVectorWidth))
    return getIntFromAttribute(*maxVectorWidth);
  emitNotFoundWarning(getLocation(), kMaxVectorWidth);
  return CPUTargetDeafultValueMap[kMaxVectorWidth];
}

} // namespace gc
} // namespace mlir