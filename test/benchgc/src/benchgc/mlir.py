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

import gc_mlir.ir

from gc_mlir.dialects import func

from benchgc.arg import Arg
from typing import Callable, List, Self

def init_i1o1_module(argin: Arg, argout: Arg, op_func: Callable[[gc_mlir.ir.Context, gc_mlir.ir.BlockArgument], gc_mlir.ir.OpResult]) -> gc_mlir.ir.Module:
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


def init_i2o1_module(argin0: Arg, argin1: Arg, argout: Arg, op_func: Callable[[gc_mlir.ir.Context, gc_mlir.ir.BlockArgument, gc_mlir.ir.BlockArgument], gc_mlir.ir.OpResult]) -> gc_mlir.ir.Module:
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

# calling python binding consumes a lot of time e.g. get_name()
# we need to cache some result to avoid duplicate call
class MLIRCache:
    # operand name cache
    opr: List[str]
    # result name cache
    res: List[str]
    # argument name cache
    arg: List[str] 
    # next hierarchy
    next: List[Self]

    def __init__(self):
        self.opr = []
        self.res = []
        self.arg = []
        self.next = []