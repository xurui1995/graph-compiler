################################################################################
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from benchgc.mlir.arg import MLIRArg
import gc_mlir.ir
from typing import Callable

import gc_mlir.dialects.tensor

from gc_mlir.dialects import func


def init_i1o1_module(
    argin: MLIRArg,
    argout: MLIRArg,
    op_func: Callable[
        [gc_mlir.ir.Context, gc_mlir.ir.BlockArgument], gc_mlir.ir.OpResult
    ],
) -> gc_mlir.ir.Module:
    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(
                name="entry",
                type=gc_mlir.ir.FunctionType.get(
                    inputs=[argin.get_mlir_type(ctx)],
                    results=[argout.get_mlir_type(ctx)],
                ),
            )
            f.attributes["llvm.emit_c_interface"] = gc_mlir.ir.UnitAttr.get()
            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                arg0: gc_mlir.ir.BlockArgument = f.entry_block.arguments[0]
                func.ReturnOp([op_func(ctx, arg0)])
        return module


def init_i2o1_module(
    argin0: MLIRArg,
    argin1: MLIRArg,
    argout: MLIRArg,
    op_func: Callable[
        [gc_mlir.ir.Context, gc_mlir.ir.BlockArgument, gc_mlir.ir.BlockArgument],
        gc_mlir.ir.OpResult,
    ],
) -> gc_mlir.ir.Module:
    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(
                name="entry",
                type=gc_mlir.ir.FunctionType.get(
                    inputs=[
                        argin0.get_mlir_type(ctx),
                        argin1.get_mlir_type(ctx),
                    ],
                    results=[argout.get_mlir_type(ctx)],
                ),
            )
            f.attributes["llvm.emit_c_interface"] = gc_mlir.ir.UnitAttr.get()

            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                arg0, arg1 = f.entry_block.arguments
                func.ReturnOp([op_func(ctx, arg0, arg1)])
        return module