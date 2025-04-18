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


def matmul_add_left_add(x, y, b, another):
    mm_out = paddle.matmul(x, y)
    bias_out = mm_out + b
    out = bias_out + another
    return bias_out, out


def matmul_add_right_add(x, y, b, another):
    mm_out = paddle.matmul(x, y)
    bias_out = mm_out + b
    out = another + bias_out
    return bias_out, out


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, b, another):
        outs = self.fn(x, y, b, another)
        return outs


class TestAPMatmulTernary(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float16"

        self.x_shape = [4, 65536, 128]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [128, 32]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

        self.b_shape = [32]
        self.b = paddle.randn(self.b_shape, dtype=self.dtype)
        self.b.stop_gradient = False

        self.another_shape = [4, 65536, 32]
        self.another = paddle.randn(self.another_shape, dtype=self.dtype)
        self.another.stop_gradient = False

    def eval_symbolic(self, net, use_cinn, profile):
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
            InputSpec(shape=self.y_shape, dtype=self.dtype),
            InputSpec(shape=self.b_shape, dtype=self.dtype),
            InputSpec(shape=self.another_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = utils.run_with_profile(profile, net, self.x, self.y, self.b, self.another)
        return out

    def notest_matmul_add_right_add(self):
        profile = False
        net = CINNSubGraphNet(matmul_add_right_add)
        cinn_outs = self.eval_symbolic(net, use_cinn=True, profile=profile)
        dy_outs = self.eval_symbolic(net, use_cinn=False, profile=profile)
        if not profile:
            for i, (a, b) in enumerate(zip(cinn_outs, dy_outs)):
                utils.check_result(self.dtype, a.numpy(), b.numpy())

    def test_matmul_add_left_add(self):
        profile = False
        net = CINNSubGraphNet(matmul_add_left_add)
        cinn_outs = self.eval_symbolic(net, use_cinn=True, profile=profile)
        dy_outs = self.eval_symbolic(net, use_cinn=False, profile=profile)
        if not profile:
            for i, (a, b) in enumerate(zip(cinn_outs, dy_outs)):
                utils.check_result(self.dtype, a.numpy(), b.numpy())


if __name__ == "__main__":
    unittest.main()
