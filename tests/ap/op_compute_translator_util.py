import code_gen_value_util

class ApOpLoadFromRegisterCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    register_var_name_attr = self.op_property.attributes.register_var_name
    register_var_name = register_var_name_attr.match(a_str=lambda x:x)
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      register_var_name
    )


class ApOpLoadFromGlobalCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    index_func_unique_id_attr = self.op_property.attributes.index_func_unique_id
    index_func_unique_id = index_func_unique_id_attr.match(a_str=lambda x:x)
    offset_var_name = self.index_program_translator_map.get_offset_var_name(
      index_func_unique_id=index_func_unique_id,
      mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
      mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
    )
    data_op_name = inputs[0].var_name
    print('data_op_name is: ', data_op_name)
    arg_name = mut_kernel_arg_id_registry.get_in_tensor_data_ptr_var_name(data_op_name)
    print('arg_name is: ', arg_name)
    ptr_var_name = self.kernel_arg_translator.get_use_name(arg_name)
    print('ptr_var_name is: ', ptr_var_name)
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{ptr_var_name}[{offset_var_name}]")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class ApOpStoreToRegisterCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map
    self.dtype2type_name = OrderedDict(
        [
            [PointerType.const_float_ptr, "const float*"],
            [PointerType.const_float16_ptr, "const half*"],
            [PointerType.float_ptr, "float*"],
            [PointerType.float16_ptr, "half*"],
            [DataType.float, "float"],
            [DataType.float16, "half"],
            [DataType.int64_t, "int64_t"],
        ]
    )
    self.ptr2type = OrderedDict(
        [
            ["float*", "float"],
            ["half*", "half"],
        ]
    )

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    mut_lir_code_gen_ctx.stmts.append(f"{self.get_out_var_name()} = {inputs[0].var_name};")
    glb_out = OrderedDict(
      map(lambda i: [f"out{i+1}", f"args.out_ptr_{i}"], range(20))
    )
    out_name = self.get_out_var_name()
    print('out_name: ', out_name)
    index_func_unique_id_attr = self.op_property.attributes.name if out_name != "out0" else []
    index_func_unique_id = index_func_unique_id_attr.match(a_str=lambda x:x) if out_name != "out0" else []

    mut_kernel_arg_id_registry.get_out_tensor_data_ptr_var_name('output') if out_name != "out0" else []
    output_seq_name = f"out_ptr_{mut_kernel_arg_id_registry.output_nums - 1}" if out_name != "out0" else []
    print('output_seq_name: ', output_seq_name)
    generated_kernel_arg_id_and_names = (
        mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
    )
    print('generated_kernel_arg_id_and_names: ', generated_kernel_arg_id_and_names)
    kernel_arg_id = filter(
      lambda item: item[1] == output_seq_name,
      generated_kernel_arg_id_and_names
    )[0] if out_name != "out0" else []
    print(kernel_arg_id) if out_name != "out0" else []
    dtype = kernel_arg_id[0].type if out_name != "out0" else []
    type_name = self.dtype2type_name[dtype] if out_name != "out0" else []
    data_type = self.ptr2type[type_name] if out_name != "out0" else []
    offset_var_name = self.index_program_translator_map.get_offset_var_name(
      index_func_unique_id=index_func_unique_id,
      mut_kernel_arg_id_registry=mut_kernel_arg_id_registry,
      mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
    ) if out_name != "out0" else []
    ptr_name = glb_out[out_name] if out_name != "out0" else []
    mut_lir_code_gen_ctx.stmts.append(
      f"{ptr_name}[{offset_var_name}] = static_cast<{data_type}>({out_name});"
    ) if out_name != "out0" else []
    return []

  def get_out_var_name(self):
    register_var_name_attr = self.op_property.attributes.register_var_name
    return register_var_name_attr.match(a_str=lambda x:x)


class PdOpDataCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    name = self.op_property.attributes.name.match(a_str=lambda x:x)
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      name
    )


class PdOpFullCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    value = self.op_property.attributes.value.match(a_f64=lambda x:x)
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{value}")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpCastCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map
    self.dtype2type_name = OrderedDict(
        [
            [DataType.float,   "float"],
            [DataType.float16,  "half"],
            [DataType.int32,     "int"],
            [DataType.int64, "int64_t"],
        ]
    )

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    dtype = self.op_property.attributes.dtype.match(a_dtype=lambda x:x)
    dtype_name = self.dtype2type_name[dtype]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"static_cast<{dtype_name}>({inputs[0].var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpExpCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"expf({inputs[0].var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpReluCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"({inputs[0].var_name} > 0 ? {inputs[0].var_name} : 0) ")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )

class PdOpErfCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    var_name = inputs[0].var_name
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"erf({var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )

class PdOpElementwisePowCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    exponent = inputs[1].var_name
    var_name = inputs[0].var_name
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"pow({var_name},{exponent})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )

class PdOpTanhCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    var_name = inputs[0].var_name
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"tanh({var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )

class CinnOpScaleCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    scale = self.op_property.attributes.scale.match(a_f32=lambda x:x)
    bias = self.op_property.attributes.bias.match(a_f32=lambda x:x)
    bias_after_scale = self.op_property.attributes.bias_after_scale.match(a_bool=lambda x:x)
    in_name = inputs[0].var_name
    true_str = f"{scale} * {in_name} + {bias}"
    false_str = f"{scale} * ({in_name} + {bias})"
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, true_str if bias_after_scale else false_str)
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpSubstractCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"({a.var_name} - {b.var_name})")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpAddCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{a.var_name} + {b.var_name}")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpMultiplyCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{a.var_name} * {b.var_name}")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpDivideCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"{a.var_name} / {b.var_name}")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class PdOpMaximumCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    a = inputs[0]
    b = inputs[1]
    out = self.get_out_cg_val(0)
    mut_lir_code_gen_ctx.let(out, f"(({a.var_name} >= {b.var_name}) ? ({a.var_name}) : ({b.var_name}))")
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class CinnOpYieldStoreCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    return inputs


class CinnOpBroadcastCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    return inputs

class CinnOpExpandCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    return [inputs[0]]

class CinnOpGenerateShapeCodeGen:
  def __init__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.index_program_translator_map = index_program_translator_map

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = self.get_out_cg_val(0)
    return [out]

  def get_out_cg_val(self, i):
    return code_gen_value_util.CodeGenValue(
      self.output_properties[i].type,
      f"op{self.op_property.op_index}_out{i}"
    )


class OpComputeTranslatorFactory:
  def __init__(self):
    self.op_name2class = OrderedDict([
      ["ap_op.load_from_register",  ApOpLoadFromRegisterCodeGen],
      ["ap_op.store_to_register",   ApOpStoreToRegisterCodeGen],
      ["ap_op.load_from_global",    ApOpLoadFromGlobalCodeGen],
      ["pd_op.data",                PdOpDataCodeGen],
      ["pd_op.full",                PdOpFullCodeGen],
      ["pd_op.cast",                PdOpCastCodeGen],
      ["pd_op.exp",                 PdOpExpCodeGen],
      ["pd_op.relu",                PdOpReluCodeGen],
      ["pd_op.tanh",                PdOpTanhCodeGen],
      ["pd_op.erf",                 PdOpErfCodeGen],
      ["pd_op.elementwise_pow",     PdOpElementwisePowCodeGen],
      ["cinn_op.scale",             CinnOpScaleCodeGen],
      ["pd_op.subtract",            PdOpSubstractCodeGen],
      ["pd_op.add",                 PdOpAddCodeGen],
      ["pd_op.multiply",            PdOpMultiplyCodeGen],
      ["pd_op.divide",              PdOpDivideCodeGen],
      ["pd_op.maximum",             PdOpMaximumCodeGen],
      ["cinn_op.yield_store",       CinnOpYieldStoreCodeGen],
      ["cinn_op.broadcast",         CinnOpBroadcastCodeGen],
      ["pd_op.expand",              CinnOpExpandCodeGen],
      ["cinn_op.generate_shape",    CinnOpGenerateShapeCodeGen]
    ])

  def __call__(self,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               index_program_translator_map):
    cls = self._get_class(op_property.op_name)
    return cls(
      op_property=op_property,
      input_properties=input_properties,
      output_properties=output_properties,
      kernel_arg_translator=kernel_arg_translator,
      index_program_translator_map=index_program_translator_map,
    )

  def _get_class(self, op_name):
    return self.op_name2class[op_name]
