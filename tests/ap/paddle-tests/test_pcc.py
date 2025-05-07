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

import os
import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import unittest
import utils

import paddle
from paddle.static import InputSpec
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, b):
        out = self.fn(x, y, b)
        return out


class TestPCCMatmulBinary(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "bfloat16"

        self.x_shape = [4, 65536, 128]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [128, 32]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

        self.b_shape = [32]
        self.b = paddle.randn(self.b_shape, dtype=self.dtype)
        self.b.stop_gradient = False

    def run_with_dy2st(self, profile):
        def matmul_add_relu(x, y, b):
            out = paddle.matmul(x, y)
            return paddle.nn.functional.relu(out + b)

        net = CINNSubGraphNet(matmul_add_relu)
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
            InputSpec(shape=self.y_shape, dtype=self.dtype),
            InputSpec(shape=self.b_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, False, input_spec)
        net.eval()
        out = utils.run_with_profile(profile, net, self.x, self.y, self.b)
        return out

    def run_with_pcc(self, profile):
        B = pct.DimVar(self.x_shape[0])
        M = pct.DimVar(self.x_shape[1])
        N = pct.DimVar(self.y_shape[1])
        K = pct.DimVar(self.x_shape[2])
        DType = pct.DTypeVar("T", self.dtype)

        def matmul_add_relu(
            x: pct.Tensor([B, M, K], DType),
            y: pct.Tensor([K, N], DType),
            b: pct.Tensor([N], DType),
        ):
            def epilogue(out):
                return paddle.nn.functional.relu(out + b)

            out = paddle.matmul(x, y)
            return epilogue(out)

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        fused_matmul = pcc.compile(matmul_add_relu, ap_path=parent_dir)
        out = fused_matmul(self.x, self.y, self.b)
        return out

    def test_matmul_add_relu(self):
        profile = False
        ap_out = self.run_with_pcc(profile=profile)
        dy2st_out = self.run_with_dy2st(profile=profile)
        if not profile:
            utils.check_result(self.dtype, ap_out, dy2st_out)


if __name__ == "__main__":
    unittest.main()
