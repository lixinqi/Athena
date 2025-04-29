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


def moe_zip(
    unzipped_tokens,
    zipped_expertwise_rowmap,
    expert_routemap_topk,
    unzipped_token_probs,
):
    zipped_tokens, zipped_prob_topk = paddle._C_ops._moe_zip(
        unzipped_tokens,
        zipped_expertwise_rowmap,
        expert_routemap_topk,
        unzipped_token_probs,
    )
    return zipped_tokens, zipped_prob_topk


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x1, x2, x3, x4):
        zipped_tokens, zipped_prob_topk = self.fn(x1, x2, x3, x4)
        return zipped_tokens, zipped_prob_topk


class TestAPZip(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        u_seqlen = 4
        token_len = 8
        seqlen = 3
        num_experts = 4
        topk = 8
        unzipped_tokens_data = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        self.unzipped_tokens_shape = [u_seqlen, token_len]
        self.unzipped_tokens_dtype = "bfloat16"
        self.unzipped_tokens = paddle.to_tensor(
            unzipped_tokens_data, dtype=self.unzipped_tokens_dtype
        )
        self.unzipped_tokens.stop_gradient = False

        zipped_expertwise_rowmap_data = [
            [0, 3, -1, -1],
            [-1, 1, -1, -1],
            [2, -1, -1, -1],
        ]
        self.zipped_expertwise_rowmap_shape = [seqlen, num_experts]
        self.zipped_expertwise_rowmap_dtype = "int32"
        self.zipped_expertwise_rowmap = paddle.to_tensor(
            zipped_expertwise_rowmap_data, self.zipped_expertwise_rowmap_dtype
        )
        self.zipped_expertwise_rowmap.stop_gradient = False

        routemap_topk_data = [
            [-1, -1, 0, 1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, -1, -1, -1, -1, -1, -1],
        ]
        self.expert_routemap_topk_shape = [seqlen, topk]
        self.expert_routemap_topk_dtype = "int32"
        self.expert_routemap_topk = paddle.to_tensor(
            routemap_topk_data, dtype=self.expert_routemap_topk_dtype
        )
        self.expert_routemap_topk.stop_gradient = False

        unzipped_token_probs_data = [[0.50000000], [1.0], [1.0], [0.50000000]]
        self.unzipped_token_probs_shape = [u_seqlen, 1]
        self.unzipped_token_probs_dtype = "float32"
        self.unzipped_token_probs = paddle.to_tensor(
            unzipped_token_probs_data, self.unzipped_token_probs_dtype
        )
        self.unzipped_token_probs.stop_gradient = False
        self.zipped_tokens_type = "bfloat16"
        self.zipped_prob_topk_type = "float32"

    def eval_symbolic(self, net, use_cinn, profile):
        input_spec = [
            InputSpec(
                shape=self.unzipped_tokens_shape, dtype=self.unzipped_tokens_dtype
            ),
            InputSpec(
                shape=self.zipped_expertwise_rowmap_shape,
                dtype=self.zipped_expertwise_rowmap_dtype,
            ),
            InputSpec(
                shape=self.expert_routemap_topk_shape,
                dtype=self.expert_routemap_topk_dtype,
            ),
            InputSpec(
                shape=self.unzipped_token_probs_shape,
                dtype=self.unzipped_token_probs_dtype,
            ),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        zipped_tokens, zipped_prob_topk = utils.run_with_profile(
            profile,
            net,
            self.unzipped_tokens,
            self.zipped_expertwise_rowmap,
            self.expert_routemap_topk,
            self.unzipped_token_probs,
        )
        return zipped_tokens, zipped_prob_topk

    def test_pure_zip(self):
        profile = False
        net = CINNSubGraphNet(moe_zip)
        cinn_out = self.eval_symbolic(net, use_cinn=True, profile=profile)
        dy2st_out = self.eval_symbolic(net, use_cinn=False, profile=profile)
        if not profile:
            utils.check_result(
                self.zipped_tokens_type, cinn_out[0].numpy(), dy2st_out[0].numpy(), True
            )

            utils.check_result(
                self.zipped_prob_topk_type,
                cinn_out[1].numpy(),
                dy2st_out[1].numpy(),
                True,
            )


if __name__ == "__main__":
    unittest.main()
