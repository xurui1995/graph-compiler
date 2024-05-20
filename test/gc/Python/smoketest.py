# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
# RUN: %python %s | FileCheck %s

from gc_mlir.ir import *
from gc_mlir.dialects import onednn_graph, func
from gc_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testCreatetOp
# CHECK onednn_graph.add
@run
def testCreatetOp():
    with Context() as ctx, Location.unknown():
        onednn_graph.register_dialect()
        module = Module.create()
        f32 = F32Type.get(ctx)
        tensor_type = RankedTensorType.get([128, 256], f32)
        with InsertionPoint(module.body):
            f = func.FuncOp(
                name="add",
                type=FunctionType.get(
                    inputs=[tensor_type, tensor_type], results=[tensor_type]
                ),
            )
            with InsertionPoint(f.add_entry_block()):
                arg0, arg1 = f.entry_block.arguments
                result = onednn_graph.AddOp(arg0, arg1).result
                func.ReturnOp([result])
        print(module)
