//===- Passes.td - Graph Compiler passes -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECT_GC_PASSES
#define GC_DIALECT_GC_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOneDNNGraphToLinalg : Pass<"convert-onednn-graph-to-linalg"> {
  let summary =
      "Lower the operations from the oneDNN Graph dialect into Linalg";
  let description = [{Lowers the `onednn_graph` ops to `linalg` ops.}];
  let dependentDialects = [
    "func::FuncDialect", "math::MathDialect", "arith::ArithDialect",
    "tensor::TensorDialect", "linalg::LinalgDialect", "linalgx::LinalgxDialect"
  ];
}

#ifdef GC_USE_GPU
def LinalgToXeGPU : Pass<"linalg-to-xegpu", "func::FuncOp"> {
  let summary = "Convert linalg dialect to XeGPU dialect.";
  let description = [{
    Lower linalg ops to XeGPU dialect.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "gpu::GPUDialect",
                           "xegpu::XeGPUDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "arith::ArithDialect",
                           "math::MathDialect",
                           "vector::VectorDialect"];
  let options = [
    Option<"kTile", "k-tile", "int64_t",
           /*default=*/"32",
           "GEMM tile size for reduction dimension.">,
    Option<"stages", "stages", "int64_t",
           /*default=*/"1",
           "Number of cooperative prefetch stages.">,
    ListOption<"dpasTile", "dpas-tile", "int64_t",
               "DPAS register block sizes MxNxK">,
  ];
}
#endif

def DeepTileContractionNamedOp
    : Pass<"deep-tile-contraction-named-op", "func::FuncOp"> {
  let summary = "Tile linalg contraction named operation deeply";
  let description =
      [{The pass tries to tile the linalg contraction named op deeply.}];
  let dependentDialects = [
    "func::FuncDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
  ];
}

def SinkOpIntoInnerLoop : Pass<"sink-op-into-inner-loop"> {
  let summary = "Sink operations into inner loops";
  let description = [{The pass tries to sink operations into inner loops as deep as possible to maximize the chance for outer loop optimization.
  }];
  let dependentDialects = [];
}

def MergeNestedForall : Pass<"merge-nested-forall"> {
  let summary = "Merge nested scf.forall operations";
  let description = [{The pass tries to merge nested forall operations.}];
  let dependentDialects = ["scf::SCFDialect"];
}

#endif // GC_DIALECT_GC_PASSES
