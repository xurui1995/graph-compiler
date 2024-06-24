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

import argparse
import gc_mlir.ir

from gc_mlir.dialects import func
from gc_mlir.dialects.linalg.opdsl.lang.dsl import DefinedOpCallable

from benchgc.arg import Arg
from typing import Dict, Callable, Any


def escape_var(var: str) -> str:
    return var.removeprefix("%").removeprefix("$")


# %arg0 -> %1
def init_i1o1_module(args: Dict[str, Arg], op_func: Callable[[gc_mlir.ir.Context, gc_mlir.ir.BlockArgument], gc_mlir.ir.OpResult]) -> gc_mlir.ir.Module:
    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(
                name="entry",
                type=gc_mlir.ir.FunctionType.get(
                    inputs=[args["arg0"].get_ranked_tensor_type(ctx)],
                    results=[args["1"].get_ranked_tensor_type(ctx)],
                ),
            )

            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                # input: %arg0 output: %1
                arg0: gc_mlir.ir.BlockArgument = f.entry_block.arguments[0]
                _1: gc_mlir.ir.OpResult = op_func(ctx, arg0)
                func.ReturnOp([_1])
        return module


# %arg0, %arg1 -> %1
def init_i2o1_module(args: Dict[str, Arg], op_func: Callable[[gc_mlir.ir.Context, gc_mlir.ir.BlockArgument, gc_mlir.ir.BlockArgument], gc_mlir.ir.OpResult]) -> gc_mlir.ir.Module:
    with gc_mlir.ir.Context() as ctx, gc_mlir.ir.Location.unknown():
        module = gc_mlir.ir.Module.create()
        with gc_mlir.ir.InsertionPoint(module.body):
            f = func.FuncOp(
                name="entry",
                type=gc_mlir.ir.FunctionType.get(
                    inputs=[
                        args["arg0"].get_ranked_tensor_type(ctx),
                        args["arg1"].get_ranked_tensor_type(ctx),
                    ],
                    results=[args["1"].get_ranked_tensor_type(ctx)],
                ),
            )

            with gc_mlir.ir.InsertionPoint(f.add_entry_block()):
                # input: %arg0, %arg1 output: %1
                arg0, arg1 = f.entry_block.arguments
                _1: gc_mlir.ir.OpResult = op_func(ctx, arg0, arg1)
                func.ReturnOp([_1])
        return module