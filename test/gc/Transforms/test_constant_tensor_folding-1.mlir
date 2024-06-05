// RUN: gc-opt --split-input-file -pass-pipeline="builtin.module(constant-subgraph-analysis,constant-tensor-folding)" %s | FileCheck %s

// CHECK-LABEL: func.func @entry
module {
    func.func @entry(%a: tensor<128xf32>, %b: tensor<128xf32>, %c: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes { llvm.emit_c_interface, onednn_graph.const_args = [0 : i32, 1 : i32] } {
        %c0 = arith.constant 0 : index
        cpuruntime.printf "HI%zu\n" %c0 : index
        %ax2 = tensor.empty() : tensor<128xf32>
        %2 = linalg.add ins(%a, %a : tensor<128xf32>,tensor<128xf32>) outs(%ax2 : tensor<128xf32>) -> tensor<128xf32>
        %bx2 = tensor.empty() : tensor<128xf32>
        %3 = linalg.add ins(%b, %b : tensor<128xf32>,tensor<128xf32>) outs(%bx2 : tensor<128xf32>) -> tensor<128xf32>
        %ax2pbx2 = tensor.empty() : tensor<128xf32>
        %4 = linalg.add ins(%2, %3 : tensor<128xf32>,tensor<128xf32>) outs(%ax2pbx2 : tensor<128xf32>) -> tensor<128xf32>
        %ax2mbx2 = tensor.empty() : tensor<128xf32>
        %5 = linalg.mul ins(%2, %3 : tensor<128xf32>,tensor<128xf32>) outs(%ax2mbx2 : tensor<128xf32>) -> tensor<128xf32>
        %ax2pbx2pc = tensor.empty() : tensor<128xf32>
        %6 = linalg.add ins(%4, %c : tensor<128xf32>,tensor<128xf32>) outs(%ax2pbx2pc : tensor<128xf32>) -> tensor<128xf32>
        %ax2mbx2mc = tensor.empty() : tensor<128xf32>
        %7 = linalg.mul ins(%5, %c : tensor<128xf32>,tensor<128xf32>) outs(%ax2mbx2mc : tensor<128xf32>) -> tensor<128xf32>
        return %6, %7 : tensor<128xf32>, tensor<128xf32>
    }
}

// CHECK: cpuruntime.printf
// CHECK: linalg.add
// CHECK: linalg.mul
// CHECK: func.func @fold
// CHECK: linalg.add
// CHECK: linalg.add
// CHECK: linalg.add
// CHECK: linalg.mul

// COM: expected output:
// COM: module {
// COM:   llvm.mlir.global external constant @__num_orig_num_args(3 : i32) {addr_space = 0 : i32} : i32
// COM:   llvm.mlir.global external constant @__compute_args(dense<[3, 2, 3, 4]> : tensor<4xi32>) {addr_space = 0 : i32} : !llvm.array<4 x i32>
// COM:   llvm.mlir.global external constant @__fold_args(dense<[4, 0, 1, 3, 4]> : tensor<5xi32>) {addr_space = 0 : i32} : !llvm.array<5 x i32>
// COM:   llvm.mlir.global external constant @__fold_buffer_ids(dense<[2, 0, 1]> : tensor<3xi64>) {addr_space = 0 : i32} : !llvm.array<3 x i64>
// COM:   func.func @entry(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {llvm.emit_c_interface, onednn_graph.const_args = [0 : i32, 1 : i32]} {
// COM:     %c0 = arith.constant 0 : index
// COM:     cpuruntime.printf "HI%zu\0A" %c0 : index
// COM:     %0 = tensor.empty() : tensor<128xf32>
// COM:     %1 = linalg.add ins(%arg2, %arg0 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
// COM:     %2 = tensor.empty() : tensor<128xf32>
// COM:     %3 = linalg.mul ins(%arg1, %arg0 : tensor<128xf32>, tensor<128xf32>) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
// COM:     return %1, %3 : tensor<128xf32>, tensor<128xf32>
// COM:   }
// COM:   func.func @fold(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {llvm.emit_c_interface} {
// COM:     %0 = tensor.empty() : tensor<128xf32>
// COM:     %1 = linalg.add ins(%arg0, %arg0 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
// COM:     %2 = tensor.empty() : tensor<128xf32>
// COM:     %3 = linalg.add ins(%arg1, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
// COM:     %4 = tensor.empty() : tensor<128xf32>
// COM:     %5 = linalg.add ins(%1, %3 : tensor<128xf32>, tensor<128xf32>) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
// COM:     %6 = tensor.empty() : tensor<128xf32>
// COM:     %7 = linalg.mul ins(%1, %3 : tensor<128xf32>, tensor<128xf32>) outs(%6 : tensor<128xf32>) -> tensor<128xf32>
// COM:     return %7, %5 : tensor<128xf32>, tensor<128xf32>
// COM:   }
// COM: }