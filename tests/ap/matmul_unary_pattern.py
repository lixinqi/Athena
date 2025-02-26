import abstract_drr
import access_topo_drr
import topo_drr_pass
import op_convertion_drr_pass
import low_level_ir_code_gen_ctx_util
import matmul_variadic_tpl
import ir_tools
import op_compute_translator_util
import program_translator_util
import kernel_arg_id_util
import kernel_arg_translator_util
import pir


class RemoveDataOp2DownSpiderOp2YieldOpPass(access_topo_drr.DrrPass):
    def __init__(self, data_op_name):
        self.data_op_name = pir.a_str(data_op_name)

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op.name = self.data_op_name
        o.data_op(
            [],
            [t.data_op_out]
        )
        o.down_spider_op = o.ap_native_op("ap_op.down_spider")
        o.down_spider_op(
            [t.data_op_out],
            [t.down_spider_op_out])
        o.yield_op = o.ap_native_op("cf.yield")
        o.yield_op(
            [t.down_spider_op_out],
            []
        )

    def result_pattern(self, o, t):
        pass


@abstract_drr.register_drr_pass("matmul_unary_fusion", nice=0)
class MatmulUnaryFusion(abstract_drr.DrrPass):
    def source_pattern(self, o, t):
        o.matmul_op = o.ap_native_op("pd_op.matmul")
        o.matmul_op(
            [t.input0, t.input1],
            [t.mm_out]
        )
        o.trivial_op = o.ap_trivial_fusion_op()
        o.trivial_op(
            [t.mm_out],
            [t.output]
        )

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
        o.fustion_op([t.input0, t.input1], [t.output])

    def constraint(self, o, t):
        program = ir_tools.copy_fused_ops_to_program(o.trivial_op, tensor_match_ctx=t)
        print("before-access_topo_pass", program)
        init_pass_manager = ir_tools.create_pass_manager()
        init_down_spider = topo_drr_pass.InitDownSpiderAccessTopoPass("mm_out")
        init_pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(init_down_spider)
        )
        init_pass_manager.run(program)
        print("after-init-access_topo_pass", program)
        pass_manager = ir_tools.create_pass_manager()
        pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("default"))
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(program)
        print("after-apply-access_topo_pass", program)
        pass_manager = ir_tools.create_pass_manager()
        remove_data_op2down_spider_op2yield_op_pass = (
            RemoveDataOp2DownSpiderOp2YieldOpPass(
                data_op_name="mm_out",
            )
        )
        pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(
                remove_data_op2down_spider_op2yield_op_pass
            )
        )
        pass_manager.run(program)
        print("after-remove-input-output-access_topo_pass", program)
        return program.empty()

    def _insert_load_from_global(self, program, input_names):
        init_pass_manager = ir_tools.create_pass_manager()

        def AddPass(input_name):
            ir_pass = topo_drr_pass.InitNaiveLoadFromGlobalAccessTopoPass(input_name)
            init_pass_manager.add_pass(
                ir_tools.create_access_topo_drr_one_step_pass(ir_pass)
            )

        map(AddPass, input_names)
        init_pass_manager.run(program)

    def _insert_store_to_global(self, program, output_names):
        init_pass_manager = ir_tools.create_pass_manager()
        ir_pass = topo_drr_pass.FakeDataStoreToGlobalForYieldAccessTopoPass(
            output_names
        )
        init_pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(ir_pass)
        )
        init_pass_manager.run(program)

    def _make_kernel_arg_translator(self):
        return matmul_variadic_tpl.make_kernel_arg_translator()

    def _replace_with_load_from_register(
        self, mut_program, load_ir_value_name, register_var_name
    ):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = topo_drr_pass.ReplaceWithLoadFromRegisterPass(
            name=load_ir_value_name, register_var_name=register_var_name
        )
        pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(drr_pass))
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        return mut_program

    def _replace_with_store_to_register(
        self, mut_program, store_ir_value_name, register_var_name
    ):
        pass_manager = ir_tools.create_pass_manager()
        drr_pass = topo_drr_pass.ReplaceWithStoreToRegisterPass(
            name=store_ir_value_name, register_var_name=register_var_name
        )
        pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(drr_pass))
        pass_manager.add_pass(ir_tools.create_dce_pass())
        pass_manager.run(mut_program)
        return mut_program

    def _get_program_translator(self, ctx, o, t):
        mut_program = ir_tools.copy_fused_ops_to_program(
            o.trivial_op, tensor_match_ctx=t
        )
        print("origin-program_translator", mut_program)
        self._insert_load_from_global(mut_program, input_names=["mm_out"])
        self._insert_store_to_global(mut_program, output_names=["output"])
        kernel_arg_translator = self._make_kernel_arg_translator()
        self._replace_with_load_from_register(
            mut_program, load_ir_value_name="mm_out", register_var_name="x"
        )
        self._replace_with_store_to_register(
            mut_program, store_ir_value_name="output", register_var_name="out"
        )
        print("after-insert-load-store-program_translator", mut_program)
        op_compute_translator_maker = (
            op_compute_translator_util.OpComputeTranslatorFactory()
        )
        program_translator = program_translator_util.ProgramTranslator(
            program_property=mut_program.copy_to_const_program_data(),
            kernel_arg_translator=kernel_arg_translator,
            index_program_translator_map=None,
            op_translator_maker=op_compute_translator_maker,
        )
        return program_translator

    def code_gen(self, ctx, o, t):
        program_translator = self._get_program_translator(ctx, o, t)
        mut_kernel_arg_id_registry = kernel_arg_id_util.KernelArgIdNameRegistry(
            code_gen_ctx=ctx, tensor_match_ctx=t, name_prefix=""
        )
        template_module = matmul_variadic_tpl.MatmulVariadicTemplate(
            program_translator=program_translator,
            mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
        )

        def get_symbolic_shape_args_list(sym_dim):
            return ctx.dim_expr_kernel_arg_id(sym_dim)

        input0_shape_kargs = map(
            get_symbolic_shape_args_list, t.input0.symbolic_shape_to_list()
        )
        input1_shape_kargs = map(
            get_symbolic_shape_args_list, t.input1.symbolic_shape_to_list()
        )
        return template_module.compile(
            input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
            input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
            output_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output),
            input0_shape_kargs=input0_shape_kargs,
            input1_shape_kargs=input1_shape_kargs,
        )
