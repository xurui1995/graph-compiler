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

import benchgc.util
import benchgc.arg.compare
from benchgc.arg.arg import Arg

from typing import List, Tuple

import benchgc.arg.binary as binary
import benchgc.arg.eltwise as eltwise
import benchgc.arg.matmul as matmul


onednn_module = {
    "binary": binary,
    "eltwise": eltwise,
    "matmul": matmul,
}


def set_default_fill(
    flags: argparse.Namespace, arg: Arg, arglist: List[Arg], is_return: bool
):
    if arg.fill_type != "-":
        return

    if is_return:
        arg.fill_type = "Z"
        arg.fill_param = []
        return

    for _, module in onednn_module.items():
        if flags.driver + "." + flags.case in module.op:
            module.default_fill(flags, arg, arglist)
            return
    # use N(0, 1) as default
    arg.fill_type = "N"
    arg.fill_param = ["0", "1"]


def set_default_compare(flags: argparse.Namespace, arg: Arg, arglist: List[Arg]):
    if arg.cmp_type != "-":
        return

    for _, module in onednn_module.items():
        if flags.driver + "." + flags.case in module.op:
            module.default_compare(flags, arg, arglist)
            return

    dtype: torch.dtype = benchgc.util.get_dtype(arg.dtype)
    arg.cmp_type = "P"
    if dtype.is_floating_point:
        arg.cmp_param = [str(torch.finfo(dtype).eps), "0"]
    else:
        arg.cmp_param = ["0", "0"]


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
        driver_module = onednn_module[driver]
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
        return benchgc.arg.compare.p2p(threshold, zero_percent, ref, res, verbose)
    if arg.cmp_type == "N":  # norm check
        threshold = float(arg.cmp_param[0])
        return benchgc.arg.compare.norm(threshold, ref, res, verbose)
    elif arg.cmp_type == "D" and len(arg.cmp_param) > 0:  # driver check
        driver: str = arg.cmp_param[0]
        driver_module = onednn_module[driver]
        return driver_module.compare(ref, res, verbose)
    else:
        raise Exception("invalid compare type or compare parameter")
