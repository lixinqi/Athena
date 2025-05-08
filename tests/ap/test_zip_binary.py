import pir
import abstract_drr

import zip_variadic_tpl
import kernel_arg_id_util
import program_translator_util
import op_compute_translator_util


@abstract_drr.register_drr_pass("pure_zip_fuse", nice=0)
class PureZipFuse(abstract_drr.DrrPass):

    def source_pattern(self, o, t):
        print("in source pattern")
        o.moe_zip_op = o.ap_native_op("pd_op._moe_zip")
        o.moe_zip_op(
            [
                t.unzipped_tokens,
                t.zipped_expertwise_rowmap,
                t.expert_routemap_topk,
                t.unzipped_token_probs,
            ],
            [t.zipped_tokens, t.zipped_probs_topk],
        )

    def constraint(self, o, t):
        return True

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
        o.fustion_op(
            [
                t.unzipped_tokens,
                t.zipped_expertwise_rowmap,
                t.expert_routemap_topk,
                t.unzipped_token_probs,
            ],
            [t.zipped_tokens, t.zipped_probs_topk],
        )

    def code_gen(self, ctx, o, t):
        mut_kernel_arg_id_registry = kernel_arg_id_util.KernelArgIdNameRegistry(
            code_gen_ctx=ctx, tensor_match_ctx=t, name_prefix=""
        )
        template_module = zip_variadic_tpl.ZipVariadicTemplate(
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry
        )
        return template_module.compile(
            unzipped_tokens_in_karg=ctx.in_tensor_data_ptr_kernel_arg_id(
                t.unzipped_tokens
            ),
            zipped_expertwise_rowmap_in_karg=ctx.in_tensor_data_ptr_kernel_arg_id(
                t.zipped_expertwise_rowmap
            ),
            expert_routemap_topk_in_karg=ctx.in_tensor_data_ptr_kernel_arg_id(
                t.expert_routemap_topk
            ),
            unzipped_token_probs_in_karg=ctx.in_tensor_data_ptr_kernel_arg_id(
                t.unzipped_token_probs
            ),
            zipped_tokens_out_karg=ctx.out_tensor_data_ptr_kernel_arg_id(
                t.zipped_tokens
            ),
            zipped_probs_topk_out_karg=ctx.out_tensor_data_ptr_kernel_arg_id(
                t.zipped_probs_topk
            ),
            topk_kargs=ctx.dim_expr_kernel_arg_id(
                t.expert_routemap_topk.symbolic_shape_to_list()[1]
            ),
            num_experts_kargs=ctx.dim_expr_kernel_arg_id(
                t.zipped_expertwise_rowmap.symbolic_shape_to_list()[1]
            ),
            token_length_kargs=ctx.dim_expr_kernel_arg_id(
                t.unzipped_tokens.symbolic_shape_to_list()[1]
            ),
            total_zipped_tokens_num_kargs=ctx.dim_expr_kernel_arg_id(
                t.zipped_tokens.symbolic_shape_to_list()[0]
            ),
        )
