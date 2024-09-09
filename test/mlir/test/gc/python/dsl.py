from functools import wraps

from gc_mlir.dialects import arith, func, linalg, scf, tensor
from gc_mlir.extras import types as T
from gc_mlir.ir import *


class dsl_module:
    def __init__(self, func):
        self.func = func
        with Context() as ctx, Location.unknown():
            self.ctx = ctx
            self.module = Module.create()

    def __call__(self, *args, **kwargs):
        with self.ctx, Location.unknown():
            with InsertionPoint(self.module.body):
                self.func(*args, **kwargs)
        print(self.module)
        return self.module


dsl_func = func.FuncOp.from_py_func
for_range = scf.for_


class dsl:
    def infer_output(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            op = f.__name__
            if len(kwargs.get("outs", [])) == 0:
                # do infer
                def do_infer():
                    if op in ("add", "abs", "copy", "max", "min"):
                        input = args[0]
                        return tensor.EmptyOp(input.type.shape, input.type.element_type)
                    if op == "matmul":
                        lhs, rhs = args[0], args[1]
                        return tensor.EmptyOp(
                            [lhs.type.shape[0], rhs.type.shape[1]],
                            lhs.type.element_type,
                        )

                kwargs["outs"] = [do_infer()]
            result = f(*args, **kwargs)
            return result

        return wrapper

    @infer_output
    def add(lhs, rhs, outs=[]):
        return linalg.add(lhs, rhs, outs=outs)

    @infer_output
    def max(lhs, rhs, outs=[]):
        return linalg.max(lhs, rhs, outs=outs)

    @infer_output
    def min(lhs, rhs, outs=[]):
        return linalg.min(lhs, rhs, outs=outs)

    @infer_output
    def matmul(lhs, rhs, outs=[]):
        return linalg.matmul(lhs, rhs, outs=outs)

    @infer_output
    def abs(input, outs=[]):
        return linalg.abs(input, outs=outs)

    @infer_output
    def copy(input, outs=[]):
        return linalg.ceil(input, outs=outs)

    def fill(value, outs=[]):
        assert len(outs) > 0, "outs must be provided"
        ele_type = outs[0].type.element_type
        return linalg.fill(arith.ConstantOp(ele_type, value), outs=outs)

    def extract_slice(result_type, input, starts, ends, strides, outs=[]):
        pass


@dsl_module
def example_1():
    @dsl_func(
        T.tensor(32, 32, T.f32()), T.tensor(32, 32, T.f32()), T.tensor(32, 32, T.f32())
    )
    def tensor_basic(t1, t2, t3):
        mul_res = dsl.matmul(t1, t2)
        return dsl.add(t3, mul_res)

m = example_1()
print(m)
# module {
#   func.func @tensor_basic(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
#     %0 = tensor.empty() : tensor<32x32xf32>
#     %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
#     %2 = tensor.empty() : tensor<32x32xf32>
#     %3 = linalg.add ins(%arg2, %1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
#     return %3 : tensor<32x32xf32>
#   }
# }
