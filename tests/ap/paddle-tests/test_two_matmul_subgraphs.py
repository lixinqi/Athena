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


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, dtype, in_features, intermedia, out_features):
        super().__init__()
        self.weight1 = paddle.randn([in_features, intermedia], dtype=dtype)
        self.bias1 = paddle.randn([intermedia], dtype=dtype)
        self.weight2 = paddle.randn([intermedia, out_features], dtype=dtype)
        self.bias2 = paddle.randn([out_features], dtype=dtype)

    def forward(self, x):
        out = paddle.matmul(x, self.weight1) + self.bias1
        out = paddle.nn.functional.relu(out)
        out = paddle.matmul(out, self.weight2) + self.bias2
        # out = paddle.nn.functional.relu(out)
        return out


class TestAPMatmulBinary(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float32"

        self.x_shape = [4, 65536, 128]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

    def eval_symbolic(self, net, use_cinn, profile):
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = utils.run_with_profile(profile, net, self.x)
        return out

    def test_eval_symbolic(self):
        profile = False
        net = CINNSubGraphNet(
            self.dtype, in_features=self.x_shape[-1], intermedia=32, out_features=128
        )
        cinn_out = self.eval_symbolic(net, use_cinn=True, profile=profile)
        d2s_out = self.eval_symbolic(net, use_cinn=False, profile=profile)
        # dy_out = utils.run_with_profile(profile, net, self.x)
        if not profile:
            utils.check_result(self.dtype, cinn_out.numpy(), d2s_out.numpy())
            # utils.check_result(self.dtype, d2s_out.numpy(), dy_out.numpy())


if __name__ == "__main__":
    unittest.main()
