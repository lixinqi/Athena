#copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.static import InputSpec
from paddle import base, tensor
from paddle.base import core
from paddle import _C_ops

def unzip(x, xscale, expert_routemap_topk, expert_prob_topk, topk, num_experts, max_tokens_per_expert):
    topk = 8
    num_experts = 4
    
    out, _, _, _ = _C_ops._moe_unzip(x, xscale, expert_routemap_topk, expert_prob_topk, max_tokens_per_expert, topk, num_experts)

    return out

def unzip_relu(x, xscale, expert_routemap_topk, expert_prob_topk, topk, num_experts, max_tokens_per_expert):
    topk = 8
    num_experts = 4
    max_tokens_per_expert = 1024
    
    out, _, _, _ = _C_ops._moe_unzip(x, xscale, expert_routemap_topk, expert_prob_topk, max_tokens_per_expert, topk, num_experts)

    return paddle.nn.functional.relu(out)

def unzip_add_relu(x, xscale, expert_routemap_topk, expert_prob_topk, topk, num_experts, max_tokens_per_expert, b):
    topk = 8
    num_experts = 4
    
    out, _, _, _ = _C_ops._moe_unzip(x, xscale, expert_routemap_topk, expert_prob_topk, max_tokens_per_expert, topk, num_experts)

    return paddle.nn.functional.relu(out + b)

class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, xs, ert, ept, max_tokens_per_expert):
        topk = 8
        num_experts = 4
        out = self.fn(x, xs, ert, ept, topk, num_experts, max_tokens_per_expert)
        return out

class TestAPUnzipBinary(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):       
        self.dtype = 'bfloat16'

        self.topk = 8
        self.num_experts = 4
        self.max_tokens_per_expert = paddle.to_tensor(1024).astype("int32")

        self.x_shape = [512, 2048]
        self.x = paddle.randn(self.x_shape, dtype=self.dtype)

        self.xscale_shape = [512, 16]
        xs_np = np.random.normal(loc=1.0, scale=0.2, size=self.xscale_shape).astype("float32")
        xs_np = np.clip(xs_np, 0.1, 3.0)
        self.xscale = paddle.to_tensor(xs_np).astype("float32")

        self.expert_routemap_topk_shape = [512, 8]
        ert_np = np.random.randint(-1, 4, self.expert_routemap_topk_shape).astype("int32")
        self.expert_routemap_topk = paddle.to_tensor(ert_np).astype("int32")
        
        self.expert_prob_topk_shape = [512, 8]
        expt_np = np.random.uniform(low=0.0, high=1.0, size=(512, 8)).astype("float32")
        self.expert_prob_topk = paddle.to_tensor(ert_np).astype("float32")

        self.b_shape = [2048]
        self.b = paddle.randn(self.b_shape, dtype=self.dtype)
        self.b.stop_gradient = False

    def eval_symbolic(self, net, use_cinn, profile):
        input_spec = [
            InputSpec(shape=self.x_shape, dtype="bfloat16"),
            InputSpec(shape=self.xscale_shape, dtype="float32"),
            InputSpec(shape=self.expert_routemap_topk_shape, dtype="int32"),
            InputSpec(shape=self.expert_prob_topk_shape, dtype="float32"),
            InputSpec(shape=[], dtype="int32")
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = utils.run_with_profile(profile, net, self.x, self.xscale, self.expert_routemap_topk, self.expert_prob_topk, self.max_tokens_per_expert)
        return out

    def test_unzip_relu(self):
        profile = False
        net = CINNSubGraphNet(unzip_relu)
        cinn_out = self.eval_symbolic(net, use_cinn=True, profile=profile)
        print(f'{cinn_out.numpy().shape=}')
        print(f'{cinn_out=}')
        # dy2st_out = self.eval_symbolic(net, use_cinn=False, profile=profile)
        # if not profile:
        #     utils.check_result(self.dtype, cinn_out.numpy(), dy2st_out.numpy())


if __name__ == "__main__":
    unittest.main()

