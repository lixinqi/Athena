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


class FacadeQuantOp(pcc.ap.FacadeOp):
    def __init__(self):
        super().__init__()

    def custom_op_name(self) -> str:
        return "ap_custom_op.facade_quant"

    def infer_meta(self) -> str:
        return "facade_utils.quant_infer_meta"

    def infer_symbolic(self) -> str:
        return "facade_utils.quant_infer_symbolic"

    def num_inputs(self) -> int:
        return 1

    def num_outputs(self, args) -> int:
        return 2  # len(args)

    def attributes_schema(self):
        # annotations matter.
        pass


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.tie_op = pcc.ap.TieOp()
        self.quant_x_op = FacadeQuantOp()
        self.quant_y_op = FacadeQuantOp()

    def forward(self, x, y):
        tie_out0, tie_out1 = self.tie_op([x, y])
        x_quanted, x_scale = self.quant_x_op([tie_out0])
        y_quanted, y_scale = self.quant_y_op([tie_out1])
        return x_quanted, x_scale, y_quanted, y_scale


class TestQuantHorizontalFusion(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.dtype = "float32"

        self.x_shape = [4, 32, 128]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)
        self.x.stop_gradient = False

        self.y_shape = [128, 64]
        self.y = paddle.randn(self.y_shape, dtype=self.dtype)
        self.y.stop_gradient = False

    def run_with_dy2st(self, use_cinn, profile):
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=self.x_shape, dtype=self.dtype),
            InputSpec(shape=self.y_shape, dtype=self.dtype),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        outs = net(self.x, self.y)
        return outs

    def run_with_pcc(self, profile):
        B = pct.DimVar(self.x_shape[0])
        M = pct.DimVar(self.x_shape[1])
        N = pct.DimVar(self.y_shape[1])
        K = pct.DimVar(self.x_shape[2])
        DType = pct.DTypeVar("T", self.dtype)

        def do_horizontal_quant(
            x: pct.Tensor([B, M, K], DType),
            y: pct.Tensor([K, N], DType),
        ):
            tie_op = pcc.ap.TieOp()
            quant_x_op = FacadeQuantOp()
            quant_y_op = FacadeQuantOp()

            tie_out0, tie_out1 = tie_op([x, y])
            # with pcc.fuse.horizontal_component():
            x_quanted, x_scale = quant_x_op([tie_out0])
            y_quanted, y_scale = quant_y_op([tie_out1])
            return x_quanted, x_scale, y_quanted, y_scale

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        fused = pcc.compile(do_horizontal_quant, ap_path=parent_dir)
        out = fused(self.x, self.y)
        return out

    def test_matmul_add_relu(self):
        profile = False
        # pcc_outs = self.run_with_pcc(profile=profile)
        dy2st_outs = self.run_with_dy2st(use_cinn=True, profile=profile)
        # if not profile:
        #    utils.check_result(self.dtype, pcc_outs, dy2st_outs)


if __name__ == "__main__":
    unittest.main()
