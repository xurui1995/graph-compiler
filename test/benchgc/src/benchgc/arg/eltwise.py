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

import torch
import benchgc.util
from benchgc.arg.arg import Arg
from benchgc.arg.compare import p2p
import argparse
from typing import List, Tuple, Set


# params format: [alg, alpha, beta]

# op should use this filling

op: Set[str] = set(["linalg.abs", "linalg.negf", "linalg.exp"])


def default_fill(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    if arg.index > 0:
        raise Exception("eltwise fill: dst filling is not allowed")
    arg.fill_param = ["eltwise", flags.case]
    if flags.driver == "linalg" and flags.case in ["abs", "exp"]:
        arg.fill_param.extend(["", ""])
    elif flags.driver == "linalg" and flags.case in ["negf"]:
        arg.fill_param.extend(["-1", "0"])
    arg.fill_type = "D"


def fill(shape: List[int], dtype: torch.dtype, params: List[str]) -> torch.Tensor:
    alg, alpha, beta = params
    nelems = benchgc.util.nelem(shape)

    float_limit: torch.finfo = torch.finfo(torch.float32)

    alpha = 0.0 if alpha == "" else float(alpha)
    beta = 0.0 if beta == "" else float(beta)

    coeff = torch.tensor(
        [1, -1, 1, -1, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 1, 1]
    )
    bias = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 88.0, 22.0, 44.0, alpha, beta])
    rand_int_mask = torch.tensor(
        [
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    rand_uni_mask = torch.tensor(
        [
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ]
    )

    if alg == "log":
        # append more value for Log validation
        coeff = torch.cat((coeff, torch.tensor([1, 1])), dim=0)
        bias = torch.cat(
            (bias, torch.tensor([float_limit.max, float_limit.min])), dim=0
        )
        rand_int_mask = torch.cat((rand_int_mask, torch.tensor([False, False])), dim=0)
        rand_uni_mask = torch.cat((rand_uni_mask, torch.tensor([False, False])), dim=0)

    repeats: int = (nelems + coeff.nelement() - 1) // coeff.nelement()

    coeff = coeff.repeat(repeats)[:nelems]
    bias = bias.repeat(repeats)[:nelems]

    rand_int_mask = rand_int_mask.repeat(repeats)[:nelems]
    rand_int = torch.where(rand_int_mask, torch.randint(0, 10, [nelems]), 0)

    rand_uni_mask = rand_uni_mask.repeat(repeats)[:nelems]
    rand_uni = torch.where(rand_uni_mask, torch.rand(nelems) * 0.09, 0)

    value = ((rand_int + rand_uni) * coeff + bias).to(dtype=dtype)
    return value.reshape(shape)


def default_compare(
    flags: argparse.Namespace,
    arg: Arg,
    arglist: List[Arg],
):
    arg.cmp_type = "D"
    arg.cmp_param = ["eltwise"]


def compare(
    ref: torch.Tensor, res: torch.Tensor, verbose: int
) -> Tuple[bool, bool | None]:
    dtype = ref.dtype
    ref = ref.to(torch.float)
    res = res.to(torch.float)
    return p2p(
        4e-6 if dtype == torch.float else benchgc.util.get_eps(dtype),
        65 if dtype.is_signed else 99.0,
        ref,
        res,
        verbose,
    )
