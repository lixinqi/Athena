# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import unittest
import utils

import paddle
from paddle.static import InputSpec
import paddle.incubate.cc as pcc


def matmul_add_relu(x, y, b):
    out = paddle.matmul(x, y)
    return paddle.nn.functional.relu(out + b)


def matmul_add_gelu(x, y, b):
    out = paddle.matmul(x, y)
    return paddle.nn.functional.gelu(out + b, False)


class FacadeMatmulOp(pcc.ap.FacadeOp):
    def __init__(self):
        super().__init__()

    def custom_op_name(self) -> str:
        return "ap_custom_op.facade_matmul"

    def infer_meta(self) -> str:
        return "facade_matmul_utils.infer_meta"

    def infer_symbolic(self) -> str:
        return "facade_matmul_utils.infer_symbolic"

    def num_inputs(self) -> int:
        return 2

    def num_outputs(self, args) -> int:
        return len(args)

    def attributes_schema(self):
        # annotations matter.
        pass


def facade_matmul_add(x, y, b):
    facade_matmul_op = FacadeMatmulOp()
    out = facade_matmul_op([x, y])
    return out[0] + b


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, b):
        out = self.fn(x, y, b)
        return out


class TestAPMatmulBinary(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float16"

        self.x_shape = [4, 64, 64]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [64, 64]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

        self.b_shape = [64]
        self.b = paddle.randn(self.b_shape, dtype=self.dtype)
        self.b.stop_gradient = False

    def eval_symbolic(self, net, use_cinn, profile):
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
            InputSpec(shape=self.y_shape, dtype=self.dtype),
            InputSpec(shape=self.b_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = utils.run_with_profile(profile, net, self.x, self.y, self.b)
        return out

    def test_matmul_add_relu(self):
        profile = False
        # net = CINNSubGraphNet(matmul_add_relu)
        net = CINNSubGraphNet(facade_matmul_add)
        cinn_out = self.eval_symbolic(net, use_cinn=True, profile=profile)
        # dy2st_out = self.eval_symbolic(net, use_cinn=False, profile=profile)
        # if not profile:
        #    utils.check_result(self.dtype, cinn_out, dy2st_out)

    def notest_matmul_add_gelu(self):
        profile = True
        net = CINNSubGraphNet(matmul_add_gelu)
        cinn_out = self.eval_symbolic(net, use_cinn=True, profile=profile)
        # dy2st_out = self.eval_symbolic(net, use_cinn=False, profile=profile)
        # if not profile:
        #    utils.check_result(self.dtype, cinn_out, dy2st_out)


if __name__ == "__main__":
    unittest.main()
