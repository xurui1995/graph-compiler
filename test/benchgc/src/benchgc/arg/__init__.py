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
import torch
import numpy
from typing import List, Tuple, Callable

# only python 3.11 support
# from typing import Self
import benchgc.util
import importlib
import gc_mlir.ir
import gc_mlir.dialects.tensor


class Arg:
    dtype: str
    shape: List[int]

    fill_type: str
    fill_param: List[str]

    cmp_type: str
    cmp_param: List[str]

    index: int
    scalar: bool

    def __init__(self, index: int):
        self.dtype = ""
        self.fill_type = "-"
        self.fill_param = []
        self.cmp_type = "-"
        self.cmp_param = []
        self.index = index

    def print_verbose(self, verbose: int):
        if verbose >= benchgc.util.ARG_VERBOSE:
            print(
                "arg{} shape: {} dtype: {} fill_type: {} fill_param: {} cmp_type: {} cmp_param: {}".format(
                    self.index,
                    self.shape,
                    self.dtype,
                    self.fill_type,
                    self.fill_param,
                    self.cmp_type,
                    self.cmp_param,
                )
            )

    # md format:
    # 0d memref/tensor: 0xf32
    # nd memref/tensor: 2x3xf32
    # scalar: f32
    def set_md(self, md: str):
        splited: List[str] = md.split("x")
        self.dtype = splited[-1]
        self.shape = []

        for dim in splited[:-1]:
            self.shape.append(int(dim))
        self.set_scalar()

    def set_scalar(self):
        # use 0xf32 to represent memref<f32>
        # use f32 to represent f32
        if self.shape == [0]:
            self.shape = []
            self.scalar = False
        elif self.shape == []:
            self.scalar = True
        else:
            self.scalar = False

    def set_fill(self, fill: str):
        splited: List[str] = fill.split(":")
        self.fill_type = splited[0]
        self.fill_param = splited[1:]

    def set_cmp(self, cmp: str):
        splited: List[str] = cmp.split(":")
        self.cmp_type = splited[0]
        self.cmp_param = splited[1:]

    def get_mlir_dtype(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.Type:
        if self.dtype == "f32":
            return gc_mlir.ir.F32Type.get(ctx)
        elif self.dtype == "f64":
            return gc_mlir.ir.F64Type.get(ctx)
        elif self.dtype == "f16":
            return gc_mlir.ir.F16Type.get(ctx)
        elif self.dtype == "bf16":
            return gc_mlir.ir.BF16Type.get(ctx)
        elif self.dtype == "u8":
            return gc_mlir.ir.IntegerType.get_unsigned(8, ctx)
        elif self.dtype == "s8":
            return gc_mlir.ir.IntegerType.get_signed(8, ctx)
        elif self.dtype == "boolean":
            return gc_mlir.ir.IntegerType.get_unsigned(1, ctx)
        elif self.dtype == "f8_e4m3":
            return gc_mlir.ir.Float8E4M3FNType.get(ctx)
        elif self.dtype == "f8_e5m2":
            return gc_mlir.ir.Float8E5M2Type.get(ctx)
        elif self.dtype == "s32":
            return gc_mlir.ir.IntegerType.get_signed(32, ctx)
        else:
            raise Exception("data type not support: %s" % self.dtype)

    def get_mlir_type(self, ctx: gc_mlir.ir.Context) -> gc_mlir.ir.Type:
        if self.shape == []:
            return self.get_mlir_dtype(ctx)
        else:
            return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))

    def get_ranked_tensor_type(
        self, ctx: gc_mlir.ir.Context
    ) -> gc_mlir.ir.RankedTensorType:
        return gc_mlir.ir.RankedTensorType.get(self.shape, self.get_mlir_dtype(ctx))

    def get_empty_op(self, ctx: gc_mlir.ir.Context) -> gc_mlir.dialects.tensor.EmptyOp:
        if self.shape == []:
            raise Exception("shape is unknown")
        return gc_mlir.dialects.tensor.EmptyOp(self.shape, self.get_mlir_dtype(ctx))

    def set_default_fill_param(
        self, flags: argparse.Namespace, args, is_return_arg: bool  # List[Self]
    ):
        if self.dtype == "":
            raise Exception("arg%d filling: dtype is not set" % self.index)
        if is_return_arg:
            if self.fill_type != "-":
                raise Exception(
                    "arg%d filling: invalid data filling set on a return arg"
                    % self.index
                )
            # set as zero filling
            self.fill_type = "Z"
            self.fill_param = []
            return

        if self.fill_type != "-":
            # no need to handle filling type if it is not default
            return

        if flags.driver in ["linalg"]:
            if flags.case in ["add", "div", "mul"] and self.index in [0, 1]:
                self.fill_type = "D"
                # fill src0 / src1
                self.fill_param = [
                    "binary",
                    "src0" if self.index == 0 else "src1",
                    args[0].dtype,
                    args[1].dtype,
                    args[2].dtype,
                ]
            elif flags.case in [
                "batch_matmul",
                "batch_matmul_transpose_a",
                "batch_matmul_transpose_b",
                "batch_matvec",
                "batch_mmt4d",
                "batch_reduce_matmul",
                "batch_vecmat",
                "matmul_transpose_b",
            ] and self.index in [0, 1]:
                self.fill_type = "D"
                self.fill_param = [
                    "matmul",
                    "src" if self.index == 0 else "wei",
                    args[0].dtype,
                    args[1].dtype,
                    args[2].dtype,
                ]
                # for matmul, the result will be amplified by k
                # so we need to find the k from the shape and append to the param to limit the fill range
                if (
                    flags.case == "batch_matmul_transpose_a"
                    and self.index == 0
                    or flags.case == "batch_matmul_transpose_b"
                    or flags.case == "matmul_transpose_b"
                    and self.index == 1
                    or flags.case == "batch_matmul"
                    and self.index == 0
                    or flags.case == "batch_matvec"
                    or flags.case == "batch_vecmat"
                    and self.index == 0
                ):
                    self.fill_type = "D"
                    self.fill_param.append(str(self.shape[-1]))

                elif flags.case == "batch_reduce_matmul" and self.index == 0:
                    self.fill_type = "D"
                    self.fill_param.append(str(self.shape[0] * self.shape[-1]))
                elif flags.case == "batch_reduce_matmul" and self.index == 1:
                    self.fill_type = "D"
                    self.fill_param.append(str(self.shape[0] * self.shape[-2]))
                elif flags.case == "batch_mmt4d":
                    self.fill_type = "D"
                    self.fill_param.append(str(self.shape[-1] * self.shape[-3]))
                else:
                    self.fill_type = "D"
                    self.fill_param.append(str(self.shape[-2]))
            elif flags.case in ["abs", "negf", "exp"]:
                self.fill_type = "D"
                self.fill_param = ["eltwise", flags.case]
                if flags.case in ["abs", "exp"]:
                    self.fill_param.extend(["", ""])
                elif flags.case == "negf":
                    self.fill_param.extend(["-1", "0"])
        if self.fill_type == "-":
            # fall back to a default normal distribution filling
            self.fill_type = "N"
            self.fill_param = ["0", "1"]

    def set_default_compare_param(
        self,
        flags: argparse.Namespace,
        args,  # List[Self],
    ):
        if self.dtype == "":
            raise Exception("arg%d compare: dtype is not set" % self.index)
        if self.cmp_type != "-":
            # no need to handle compare type if it is not default
            return

        if flags.driver in ["linalg"]:
            if flags.case in ["add", "div", "mul"]:
                self.cmp_type = "D"
                self.cmp_param = ["binary"]
            elif flags.case in [
                "batch_matmul",
                "batch_matmul_transpose_a",
                "batch_matmul_transpose_b",
                "batch_matvec",
                "batch_mmt4d",
                "batch_reduce_matmul",
                "batch_vecmat",
                "matmul_transpose_b",
            ]:
                self.cmp_type = "D"
                self.cmp_param = ["matmul"]
            elif flags.case in ["abs", "negf", "exp"]:
                self.cmp_type = "D"
                self.cmp_param = ["eltwise"]

        if self.cmp_type == "-":
            # p2p check with a default threshold based on data type
            # do not check mistrust
            dtype: torch.dtype = benchgc.util.get_dtype(self.dtype)
            self.cmp_type = "P"
            if dtype.is_floating_point:
                self.cmp_param = [str(torch.finfo(dtype).eps), "0"]
            else:
                self.cmp_param = ["0", "0"]


def fill_tensor(flags: argparse.Namespace, arg: Arg, idx: int) -> torch.Tensor:
    if arg.dtype == "" or arg.fill_type == "":
        raise Exception("arg%d filling: dtype/fill_type is not set" % idx)

    if arg.fill_type == "N" and len(arg.fill_param) == 2:
        # Normal distribution
        mean = float(arg.fill_param[0])
        std = float(arg.fill_param[1])
        tensor = torch.normal(mean=mean, std=std, size=arg.shape)

    elif arg.fill_type == "P" and len(arg.fill_param) == 1:
        # Poisson distribution
        _lambda = float(arg.fill_param[0])
        lambda_tensor = torch.full(arg.shape, _lambda)
        tensor = torch.poisson(lambda_tensor)
    elif arg.fill_type == "B" and len(arg.fill_param) == 2:
        # Binomial distribution
        n = int(arg.fill_param[0])
        p = float(arg.fill_param[1])
        bdist = torch.distributions.binomial.Binomial(total_count=n, probs=p)
        tensor = bdist.sample(torch.Size(arg.shape))
    elif arg.fill_type == "U" and len(arg.fill_param) == 2:
        # Uniform distribution
        a = float(arg.fill_param[0])
        b = float(arg.fill_param[1])
        tensor = torch.distributions.uniform.Uniform(a, b).sample(torch.Size(arg.shape))
    elif arg.fill_type == "F" and len(arg.fill_param) == 1:
        # read from pytorch tensor dump file
        filename = arg.fill_param[0]
        tensor = torch.load(f=filename)
        if not isinstance(tensor, torch.Tensor):
            raise Exception(
                "torch object from file %s is not a tensor object" % filename
            )
        if tensor.shape != torch.Size(arg.shape):
            raise Exception(
                "tensor object from file %s does not match shape" % filename
            )
        if tensor.dtype != benchgc.util.get_dtype(arg.dtype):
            raise Exception(
                "tensor object from file %s does not match dtype" % filename
            )
    elif arg.fill_type == "D" and len(arg.fill_param) > 0:
        # Driver fill
        driver: str = arg.fill_param[0]
        driver_module = importlib.import_module("benchgc.arg.%s" % driver)
        tensor = driver_module.fill(
            arg.shape, benchgc.util.get_dtype(arg.dtype), arg.fill_param[1:]
        )
    elif arg.fill_type == "Z":
        tensor = torch.zeros(size=arg.shape)
    else:
        raise Exception("invalid fill type or fill parameter")

    tensor = tensor.to(benchgc.util.get_dtype(arg.dtype))
    if flags.verbose >= benchgc.util.INPUT_VERBOSE:
        print("fill arg%d: " % idx)
        print(tensor)
    return tensor


def compare_tensor(
    arg: Arg, ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:

    if arg.cmp_type == "P":  # p2p check
        threshold = float(arg.cmp_param[0])
        zero_percent = float(arg.cmp_param[1])
        return p2p(threshold, zero_percent, ref, res, verbose)
    if arg.cmp_type == "N":  # norm check
        threshold = float(arg.cmp_param[0])
        return norm(threshold, ref, res, verbose)
    elif arg.cmp_type == "D" and len(arg.cmp_param) > 0:  # driver check
        driver: str = arg.cmp_param[0]
        driver_module = importlib.import_module("benchgc.arg.%s" % driver)
        return driver_module.compare(ref, res, verbose)
    else:
        raise Exception("invalid compare type or compare parameter")


def iterate_tensor(tensor: torch.Tensor, fn: Callable[[Tuple[int, ...]], None]):
    if tensor.ndim == 0:
        fn(tuple())
        return
    index: List[int] = [0] * tensor.ndim

    def dfs(depth: int):
        if depth == tensor.ndim:
            fn(tuple(index))
        else:
            for i in range(tensor.shape[depth]):
                index[depth] = i
                dfs(depth + 1)

    dfs(0)


def norm(
    threshold: float, ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:

    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)
    if f32_ref.nelement() == 0:
        return (True, None)

    diff_square_sum = torch.square(torch.subtract(f32_ref, f32_res)).sum()
    square_sum = torch.square(f32_ref).sum()

    l2_diff_norm = torch.sqrt(diff_square_sum / square_sum).item()
    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print("norm check: %.10f / threshold: %.10f" % (l2_diff_norm, threshold))

    return (l2_diff_norm < threshold, None)


def p2p(
    threshold: float,
    zero_percent: float,
    ref: torch.Tensor,
    res: torch.Tensor,
    verbose: int,
) -> Tuple[bool, bool | None]:

    if verbose >= benchgc.util.COMPARE_VERBOSE:
        print("p2p check: threshold: %.7f" % threshold)
    f32_ref = ref.to(torch.float32)
    f32_res = res.to(torch.float32)

    check = torch.tensor(False)

    check = check.bitwise_or(torch.bitwise_and(f32_ref.isnan(), f32_res.isnan()))
    check = check.bitwise_or(torch.bitwise_and(f32_ref.isneginf(), f32_res.isneginf()))
    check = check.bitwise_or(torch.bitwise_and(f32_ref.isposinf(), f32_res.isposinf()))

    # choose diff/rel_diff based on value
    abs_diff = (f32_ref - f32_res).abs()
    rel_diff = abs_diff / torch.where(
        f32_ref.abs() > numpy.finfo(numpy.float32).smallest_subnormal,
        f32_ref.abs(),
        1,
    )
    # pick a diff for comparison
    diff = torch.where(f32_ref.abs() > 1e-5, rel_diff, abs_diff)

    check = check.bitwise_or(diff <= threshold)

    if verbose >= benchgc.util.OUTPUT_VERBOSE:
        iterate_tensor(
            check,
            lambda idx: print(
                "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                % (
                    idx,
                    f32_ref[idx].item(),
                    f32_res[idx].item(),
                    abs_diff[idx].item(),
                    rel_diff[idx].item(),
                )
            ),
        )
    if check.all():
        # check mistrusted
        zero = res.nelement() - res.count_nonzero().item()
        if res.nelement() < 10:
            mistrust = False
        else:
            mistrust = zero * 100.0 / res.nelement() > zero_percent
        return (True, mistrust)
    else:
        if (
            verbose < benchgc.util.OUTPUT_VERBOSE
        ):  # skip verbose print if full output tensor is alrady printed
            fail = torch.argwhere(torch.where(check, 0, 1))
            if verbose < benchgc.util.ERROR_OUTPUT_VERBOSE:
                # only print top 10 failed data points if verbose level does not satisfied
                fail = fail[:10]
            for idx in fail:
                index: Tuple[int, ...] = tuple(idx.tolist())
                print(
                    "%20s: ref: %12.7f res: %12.7f abs_diff: %12.7f rel_diff: %12.7f"
                    % (
                        index,
                        f32_ref[index].item(),
                        f32_res[index].item(),
                        abs_diff[index].item(),
                        rel_diff[index].item(),
                    )
                )
        return (False, None)
