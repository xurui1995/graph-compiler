// RUN: gc-opt %s --early-dispatch-microkernel --convert-microkernel-to-dnnl-func --microkernel-invariant-code-motion --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata --lower-affine -finalize-memref-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-complex-to-llvm --canonicalize --cse --reconcile-unrealized-casts --symbol-dce | gc-cpu-runner -e main -entry-point-result=void

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_brgemm() {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c4_index = arith.constant 4 : index
    %c8_index = arith.constant 8 : index
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x32x32xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x32x2xbf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<4x8x32x32xf32>
    scf.for %arg0 = %c0_index to %c4_index step %c1_index {
        scf.for %arg1 = %c0_index to %c8_index step %c1_index {
	      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
	      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32x32xf32>)
	      %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	      %subview_4 = memref.subview %alloc_0[%arg1, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	      %0 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (bf16, bf16) 
	      microkernel.brgemm.prologue(%0) : (i64) -> ()
	      microkernel.brgemm(%0, %subview, %subview_4, %alloc_3, %c16_i64, %c0_i64) : (i64, memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	      microkernel.brgemm.epilogue(%0) : (i64) -> ()
	      %subview_5 = memref.subview %alloc_1[%arg0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3, %subview_5 : memref<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%alloc_3 : memref<32x32xf32>) {
	      ^bb0(%in: f32, %in_7: f32, %out: f32):
		%1 = arith.addf %in, %in_7 : f32
		linalg.yield %1 : f32
	      }
	      %subview_6 = memref.subview %alloc_2[%arg0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3 : memref<32x32xf32>) outs(%subview_6 : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
	      ^bb0(%in: f32, %out: f32):
		%1 = arith.maximumf %in, %cst : f32
		linalg.yield %1 : f32
	      }
	      memref.dealloc %alloc_3 : memref<32x32xf32>
        }
	%alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<32x32xf32>)
	%subview_7 = memref.subview %alloc[%c0_index, 0, 0, 0] [1, 2, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xbf16> to memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
	%subview_8 = memref.subview %alloc_0[%c0_index, 0, 0, 0, 0] [1, 2, 16, 32, 2] [1, 1, 1, 1, 1] : memref<8x16x16x32x2xbf16> to memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
	%10 = microkernel.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (stride) data_type = (bf16, bf16) 
	microkernel.brgemm.prologue(%10) : (i64) -> ()
	microkernel.brgemm(%10, %subview_7, %subview_8, %alloc_4, %c2_i64, %c0_i64) : (i64, memref<2x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<2x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xf32>, i64, i64) -> ()
	microkernel.brgemm.epilogue(%10) : (i64) -> ()
    }
    return
  }

  func.func @main() {
    call @simple_brgemm() : ()->()
    // COM: parallelcpu.printf "BRGEMM DONE\n"
    return
  }

  // COM: CHECK: BRGEMM DONE
}