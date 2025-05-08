import abstract_drr
import access_topo_drr
import topo_drr_pass
import op_convertion_drr_pass
import unzip_tpl
import matmul_epilogue_pass
import ir_tools
import index_program_translator_util
import op_compute_translator_util
import program_translator_util
import kernel_arg_id_util
import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util
import pir
import umprime

@abstract_drr.register_drr_pass("moeunzip_fusion", nice=0)
class MoeUnzipBinaryFusion(abstract_drr.DrrPass):
  def source_pattern(self, o, t):
    in_num = self.number_of_inputs()
    out_num = self.number_of_outputs()
    o.unzip_op = o.ap_native_op("pd_op._moe_unzip")
    o.unzip_op(
        [t.input0, t.input1, t.input2, t.input3, t.input4],
        [t.unzip_out0, t.unzip_out1, t.unzip_out2, t.unzip_out3, t.unzip_out4]
    )

    o.trivial_op = o.ap_trivial_fusion_op()
    o.trivial_op(
        [t.unzip_out0],
        [t.output]
    )

  def result_pattern(self, o, t):
    o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
    o.fustion_op([t.input0, t.input1, t.input2, t.input3, t.input4], [t.output, t.unzip_out0, t.unzip_out1, t.unzip_out2, t.unzip_out3, t.unzip_out4])

  def constraint(self, o, t):
    return True

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
    ir_pass = topo_drr_pass.FakeDataStoreToGlobalForYieldAccessTopoPass(output_names)
    init_pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(ir_pass))
    init_pass_manager.run(program)

  def _make_kernel_arg_translator(self):
    return unzip_tpl.make_kernel_arg_translator()

  def _apply_topo_access_passes(self, mut_program, anchor_data_op_name):
    init_pass_manager = ir_tools.create_pass_manager()
    init_down_spider = topo_drr_pass.InitDownSpiderAccessTopoPass(anchor_data_op_name)
    init_pass_manager.add_pass(
        ir_tools.create_access_topo_drr_one_step_pass(init_down_spider)
    )
    init_pass_manager.run(mut_program)
    pass_manager = ir_tools.create_pass_manager()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("default"))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(mut_program)

  def _simplify_index_program(self, mut_program):
    pass_manager = ir_tools.create_pass_manager()
    drr_pass = topo_drr_pass.ConvertUpSpiderStoreDataOpToYieldOpPass()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(drr_pass))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(mut_program)
    return mut_program

  def _make_index_func_unique_id2index_program(
          self, compute_program, anchor_data_op_name, input_names, output_names):
    full_index_program = compute_program.clone()
    self._apply_topo_access_passes(full_index_program, anchor_data_op_name)
    def MatchAndCopyInputIndex(dst_input_name):
        pass_manager = ir_tools.create_pass_manager()
        removed_programs = MutableList()
        rm_elementwise_drr_pass = matmul_epilogue_pass.RemoveElementInputIndexPass(
            src_data_op_name=anchor_data_op_name,
            dst_load_from_global_op_name=dst_input_name
        )
        rm_elementwise_ir_pass = ir_tools.create_access_topo_drr_one_step_pass(
            rm_elementwise_drr_pass,
            matched_pattern_mut_list=removed_programs
        )
        pass_manager.add_pass(rm_elementwise_ir_pass)
        rm_broadcast_drr_pass = matmul_epilogue_pass.RemoveBroadcastInputIndexPass(
            src_data_op_name=anchor_data_op_name,
            dst_load_from_global_op_name=dst_input_name
        )
        rm_broadcast_ir_pass = ir_tools.create_access_topo_drr_one_step_pass(
            rm_broadcast_drr_pass,
            matched_pattern_mut_list=removed_programs
        )
        pass_manager.add_pass(rm_broadcast_ir_pass)
        pass_manager.run(full_index_program)
        def Converter(program):
          return [dst_input_name, self._simplify_index_program(program)]
        return map(Converter, removed_programs)
    input_and_index_programs = flat_map(MatchAndCopyInputIndex, input_names)
    def MatchAndCopyOutputIndex(dst_output_name):
        pass_manager = ir_tools.create_pass_manager()
        removed_programs = MutableList()
        drr_pass = matmul_epilogue_pass.RemoveOutputIndexPass(
            src_data_op_name=anchor_data_op_name,
            dst_store_to_global_op_name=dst_output_name
        )
        ir_pass = ir_tools.create_access_topo_drr_one_step_pass(
            drr_pass,
            matched_pattern_mut_list=removed_programs
        )
        pass_manager.add_pass(ir_pass)
        pass_manager.run(full_index_program)
        def Converter(program):
          return [dst_output_name, self._simplify_index_program(program)]
        return map(Converter, removed_programs)
    output_and_index_programs = flat_map(MatchAndCopyOutputIndex, output_names)
    return OrderedDict([*input_and_index_programs, *output_and_index_programs])

  def _replace_with_load_from_register(
      self, mut_program, load_ir_value_name, register_var_name):
    pass_manager = ir_tools.create_pass_manager()
    drr_pass = topo_drr_pass.ReplaceWithLoadFromRegisterPass(
        name=load_ir_value_name,
        register_var_name=register_var_name
    )
    pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(drr_pass))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(mut_program)
    return mut_program

  def _replace_with_store_to_register(
      self, mut_program, store_ir_value_name, register_var_name):
    pass_manager = ir_tools.create_pass_manager()
    drr_pass = topo_drr_pass.ReplaceWithStoreToRegisterPass(
        name=store_ir_value_name,
        register_var_name=register_var_name
    )
    pass_manager.add_pass(ir_tools.create_access_topo_drr_one_step_pass(drr_pass))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(mut_program)
    return mut_program

  def _get_program_translator(self, ctx, o, t):
    mut_program = ir_tools.copy_fused_ops_to_program(
      o.trivial_op, tensor_match_ctx=t
    )
    print("before all pass mut_program :\n", mut_program)
    pass_manager = ir_tools.create_pass_manager()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("umprime"))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(mut_program)
    print("after-umprime:\n", mut_program)
    self._insert_load_from_global(
      mut_program,
      input_names=["unzip_out0"]
    )
    self._insert_store_to_global(
      mut_program,
      output_names=["output"]
    )
    print("after-insert_load_and_store:\n", mut_program)
    kernel_arg_translator = self._make_kernel_arg_translator()
    index_func_unique_id2index_program = self._make_index_func_unique_id2index_program(
      mut_program,
      anchor_data_op_name="unzip_out0",
      input_names=[],
      output_names=[],
    )
    print("index_func_unique_id2index_program:\n", index_func_unique_id2index_program)
    index_program_translator_map = index_program_translator_util.IndexProgramTranslatorMap(
      index_func_unique_id2index_program=index_func_unique_id2index_program,
      kernel_arg_translator=kernel_arg_translator,
      anchor_iter_var_names=unzip_tpl.get_anchor_iter_var_names()
    )
    self._replace_with_load_from_register(
      mut_program,
      load_ir_value_name="unzip_out0",
      register_var_name="x"
    )
    self._replace_with_store_to_register(
      mut_program,
      store_ir_value_name="output",
      register_var_name="out"
    )
    print("mut_program:", mut_program)
    op_compute_translator_maker = op_compute_translator_util.OpComputeTranslatorFactory()
    program_translator = program_translator_util.ProgramTranslator(
      program_property=mut_program.copy_to_const_program_data(),
      kernel_arg_translator=kernel_arg_translator,
      index_program_translator_map=index_program_translator_map,
      op_translator_maker=op_compute_translator_maker
    )
    return program_translator

  def code_gen(self, ctx, o, t):
    program_translator = self._get_program_translator(ctx, o, t)
    mut_kernel_arg_id_registry = kernel_arg_id_util.KernelArgIdNameRegistry(
      code_gen_ctx=ctx,
      tensor_match_ctx=t,
      name_prefix=""
    )
    template_module = unzip_tpl.MoeUnzipVariadicTemplate(
      program_translator=program_translator,
      mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
    )
    def get_symbolic_shape_args_list(sym_dim):
        return ctx.dim_expr_kernel_arg_id(sym_dim)
    input0_shape_kargs = map(get_symbolic_shape_args_list, t.input0.symbolic_shape_to_list())
    output_shape_kargs = map(get_symbolic_shape_args_list, t.output.symbolic_shape_to_list())
    print(f'{input0_shape_kargs=}')
    print(f'{output_shape_kargs=}')
    return template_module.compile(
        input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
        input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
        input2_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input2),
        input3_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input3),
        output0_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output),
        output1_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.unzip_out1),
        output2_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.unzip_out2),
        output3_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.unzip_out3),
        output4_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.unzip_out4),
        input0_shape_kargs=input0_shape_kargs,
        output_shape_kargs=output_shape_kargs,
    )

class NumberOfInputsTrait0():
  def number_of_inputs(self):
    return 0

class NumberOfInputsTrait1():
  def number_of_inputs(self):
    return 1

class NumberOfInputsTrait2():
  def number_of_inputs(self):
    return 2

class NumberOfInputsTrait3():
  def number_of_inputs(self):
    return 3

class NumberOfInputsTrait4():
  def number_of_inputs(self):
    return 4

class NumberOfInputsTrait5():
  def number_of_inputs(self):
    return 5

class NumberOfInputsTrait6():
  def number_of_inputs(self):
    return 6

class NumberOfInputsTrait7():
  def number_of_inputs(self):
    return 7

class NumberOfInputsTrait8():
  def number_of_inputs(self):
    return 8

class NumberOfInputsTrait9():
  def number_of_inputs(self):
    return 9

class NumberOfInputsTrait10():
  def number_of_inputs(self):
    return 10

class NumberOfInputsTrait11():
  def number_of_inputs(self):
    return 11

class NumberOfInputsTrait12():
  def number_of_inputs(self):
    return 12

class NumberOfInputsTrait13():
  def number_of_inputs(self):
    return 13

class NumberOfInputsTrait14():
  def number_of_inputs(self):
    return 14

class NumberOfInputsTrait15():
  def number_of_inputs(self):
    return 15

class NumberOfInputsTrait16():
  def number_of_inputs(self):
    return 16

class NumberOfInputsTrait17():
  def number_of_inputs(self):
    return 17


class NumberOfOutputsTrait0():
  def number_of_outputs(self):
    return 0

class NumberOfOutputsTrait1():
  def number_of_outputs(self):
    return 1

class NumberOfOutputsTrait2():
  def number_of_outputs(self):
    return 2

class NumberOfOutputsTrait3():
  def number_of_outputs(self):
    return 3

class NumberOfOutputsTrait4():
  def number_of_outputs(self):
    return 4

class NumberOfOutputsTrait5():
  def number_of_outputs(self):
    return 5

class NumberOfOutputsTrait6():
  def number_of_outputs(self):
    return 6

class NumberOfOutputsTrait7():
  def number_of_outputs(self):
    return 7

class NumberOfOutputsTrait8():
  def number_of_outputs(self):
    return 8

class NumberOfOutputsTrait9():
  def number_of_outputs(self):
    return 9

class NumberOfOutputsTrait10():
  def number_of_outputs(self):
    return 10

class NumberOfOutputsTrait11():
  def number_of_outputs(self):
    return 11

class NumberOfOutputsTrait12():
  def number_of_outputs(self):
    return 12

class NumberOfOutputsTrait13():
  def number_of_outputs(self):
    return 13

class NumberOfOutputsTrait14():
  def number_of_outputs(self):
    return 14

class NumberOfOutputsTrait15():
  def number_of_outputs(self):
    return 15

class NumberOfOutputsTrait16():
  def number_of_outputs(self):
    return 16

class NumberOfOutputsTrait17():
  def number_of_outputs(self):
    return 17

class NumberOfOutputsTrait18():
  def number_of_outputs(self):
    return 18

class NumberOfOutputsTrait19():
  def number_of_outputs(self):
    return 19

class NumberOfOutputsTrait20():
  def number_of_outputs(self):
    return 20

class NumberOfOutputsTrait21():
  def number_of_outputs(self):
    return 21

class NumberOfOutputsTrait22():
  def number_of_outputs(self):
    return 22
def get_mixin_class(base_class, number_of_inputs, number_of_outputs):
  num_inputs_to_input_trait_class = [
    None,
    NumberOfInputsTrait1,
    NumberOfInputsTrait2,
    NumberOfInputsTrait3,
    NumberOfInputsTrait3,
    NumberOfInputsTrait4,
    NumberOfInputsTrait5,
    NumberOfInputsTrait6,
    NumberOfInputsTrait7,
    NumberOfInputsTrait8,
    NumberOfInputsTrait9,
    NumberOfInputsTrait10,
    NumberOfInputsTrait11,
    NumberOfInputsTrait12,
    NumberOfInputsTrait13,
    NumberOfInputsTrait14,
    NumberOfInputsTrait15,
    NumberOfInputsTrait16,
    NumberOfInputsTrait17,
  ]
  num_outputs_to_output_trait_class = [
    None,
    NumberOfOutputsTrait1,
    NumberOfOutputsTrait2,
    NumberOfOutputsTrait3,
    NumberOfOutputsTrait4,
    NumberOfOutputsTrait5,
    NumberOfOutputsTrait6,
    NumberOfOutputsTrait7,
    NumberOfOutputsTrait8,
    NumberOfOutputsTrait9,
    NumberOfOutputsTrait10,
    NumberOfOutputsTrait11,
    NumberOfOutputsTrait12,
    NumberOfOutputsTrait13,
    NumberOfOutputsTrait14,
    NumberOfOutputsTrait15,
    NumberOfOutputsTrait16,
    NumberOfOutputsTrait17,
    NumberOfOutputsTrait18,
    NumberOfOutputsTrait19,
    NumberOfOutputsTrait20,
    NumberOfOutputsTrait21,
    NumberOfOutputsTrait22,
  ]
  return type(
    f"MoeUnzipEpilogueFusion{number_of_inputs}_{number_of_outputs}",
    [
      base_class,
      num_inputs_to_input_trait_class[number_of_inputs],
      num_outputs_to_output_trait_class[number_of_outputs],
    ],
    BuiltinSerializableAttrMap()
  )

def register_class(base_class, max_num_inputs, max_num_outputs):
  def register_drr_class(num_inputs, num_outputs):
    abstract_drr.register_drr_pass(f"moeunzip_binary_in{num_inputs}_out{num_outputs}_fusion", nice=0)(
      get_mixin_class(base_class, num_inputs, num_outputs)
    )

  def register_num_inputs_drr_classes(num_inputs):

    def register_num_outputs_drr_classes(num_outputs):
      return register_drr_class(num_inputs+2, num_outputs+1)

    map(register_num_outputs_drr_classes, range(max_num_outputs))
    print('done max outputs')
    return None

  map(register_num_inputs_drr_classes, range(max_num_inputs))
  return None

register_class(base_class=MoeUnzipBinaryFusion, max_num_inputs=10, max_num_outputs=10)