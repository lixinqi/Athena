import os
import unittest

import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap"

class TestMoeUnzipEpilogue(unittest.TestCase):
    def setUp(self):
        dtype = 'bfloat16'

        self.topk = 8
        self.num_experts = 4
        self.max_tokens_per_expert = 1024
        
        x_shape = [512, 2048]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = False

        xscale_shape = [512, 16]
        xs_np = np.random.normal(loc=1.0, scale=0.2, size=xscale_shape).astype("float32")
        xs_np = np.clip(xs_np, 0.1, 3.0)
        self.xscale = paddle.to_tensor(xs_np).astype("float32")
        self.xscale.stop_gradient = False

        expert_routemap_topk_shape = [512, 8]
        ert_np = np.random.randint(-1, self.num_experts, expert_routemap_topk_shape).astype("int32")
        self.expert_routemap_topk = paddle.to_tensor(ert_np).astype("int32")
        self.expert_routemap_topk.stop_gradient = False
        
        expert_prob_topk_shape = [512, 8]
        expt_np = np.random.uniform(low=0.0, high=1.0, size=expert_prob_topk_shape).astype("float32")
        self.expert_prob_topk = paddle.to_tensor(ert_np).astype("float32")
        self.expert_prob_topk.stop_gradient = False

    def getSubGraph(self):
        seqlen = pct.DimVar(512)
        token_len = pct.DimVar(2048)
        xscale_dim_1 = pct.DimVar(16)
        DType = pct.DTypeVar("T", "bfloat16")

        def foo(
            x: pct.Tensor([seqlen, token_len], DType),
            xscale: pct.Tensor([seqlen, xscale_dim_1], pct.DTypeVar("T", "float32")),
            expert_routemap_topk: pct.Tensor([seqlen, self.topk], pct.DTypeVar("T", "int32")),
            expert_prob_topk: pct.Tensor([seqlen, self.topk], pct.DTypeVar("T", "float32")),
        ):

            tmp, _, _, _, _ = paddle._C_ops._moe_unzip(x, xscale, expert_routemap_topk, expert_prob_topk, self.max_tokens_per_expert, self.topk, self.num_experts)
            tmp2 = paddle.nn.functional.relu(tmp)
            return tmp2

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        fused_foo = pcc.compile(
            foo, ap_path=f"{os.path.dirname(paddle.__file__)}/ap/"
        )
        ap_outs = fused_foo(self.x, self.xscale, self.expert_routemap_topk, self.expert_prob_topk)
        # dy_outs = foo(self.x, self.y, self.b)
        # for dy_out, ap_out in zip(dy_outs, ap_outs):
            # np.testing.assert_allclose(dy_out, ap_out, atol=1e-1)


if __name__ == "__main__":
    unittest.main()