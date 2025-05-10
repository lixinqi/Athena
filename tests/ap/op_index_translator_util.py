import index_code_gen_value_util


class PdOpDataCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.anchor_iter_var_names = anchor_iter_var_names
    self.anchor_iter_dim_splits = anchor_iter_dim_splits

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    assert len(self.anchor_iter_var_names) == len(self.anchor_iter_dim_splits), "The length of anchor_iter_var_names and anchor_iter_dim_splits is expected to be the same."
    return [index_code_gen_value_util.IndexCodeGenValue(
      self.anchor_iter_var_names,
      self.anchor_iter_dim_splits
    )]


class PdOpFullIntArrayCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.anchor_iter_var_names = anchor_iter_var_names
    self.anchor_iter_dim_splits = anchor_iter_dim_splits

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    out = index_code_gen_value_util.IndexCodeGenValue(None, None)
    def get_int64(attr):
      return attr.match(a_i64=lambda x:x)
    def convert_list(lst):
      return map(get_int64, lst)
    out.const_data = self.op_property.attributes.value.match(
      a_array=convert_list
    )
    return [out]


class PdOpSumCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.anchor_iter_var_names = anchor_iter_var_names
    self.anchor_iter_dim_splits = anchor_iter_dim_splits

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    input_iter_var_names = inputs[0].iter_var_names
    input_iter_dim_splits = inputs[0].iter_dim_splits
    def is_reduced_axes(dim):
      return False if len(filter(lambda x: int(x) == dim, inputs[1].const_data)) == 0 else True
    input_dim_split_starts = MutableList()
    input_dim_split_starts.append(0)
    def is_anchor_reduced_axes(i):
      dim_split_start = int(input_dim_split_starts[i])
      dim_split_stop = dim_split_start + input_iter_dim_splits[i]
      input_dim_split_starts.append(dim_split_stop)
      is_reduced_axes_result = filter(
        lambda dim: is_reduced_axes(dim), range(dim_split_start, dim_split_stop)
      ) 
      return False if len(is_reduced_axes_result) == 0 else True
    anchor_non_reduced_axes = filter(
      lambda i: is_anchor_reduced_axes(i) == False,
      range(len(input_iter_var_names))
    )
    output_iter_var_names = map(
      lambda i: input_iter_var_names[i],
      anchor_non_reduced_axes
    )
    output_iter_dim_splits = map(
      lambda i: input_iter_dim_splits[i],
      anchor_non_reduced_axes
    )
    return [index_code_gen_value_util.IndexCodeGenValue(
      output_iter_var_names,
      output_iter_dim_splits
    )]


class CinnOpReshapeCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.anchor_iter_var_names = anchor_iter_var_names
    self.anchor_iter_dim_splits = anchor_iter_dim_splits

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    symbolic_shape = self.input_properties[0].symbolic_shape
    input_iter_var_names = inputs[0].iter_var_names
    input_iter_dim_splits = inputs[0].iter_dim_splits
    def get_dim_var_name(i):
      dim_expr = symbolic_shape[i]
      arg_var_name = mut_kernel_arg_id_registry.get_dim_expr_var_name(dim_expr)
      return self.kernel_arg_translator.get_use_name(arg_var_name)
    input_dim_split_starts = MutableList()
    input_dim_split_starts.append(0)
    def get_anchor_iter_dims(i):
      dim_split_start = int(input_dim_split_starts[i])
      dim_split_stop = dim_split_start + input_iter_dim_splits[i]
      input_dim_split_starts.append(dim_split_stop)
      current_dim_names = map(
        lambda idx: get_dim_var_name(idx), range(dim_split_start, dim_split_stop)
      )
      return " * ".join(current_dim_names)
    rank = len(input_iter_dim_splits)
    anchor_iter_dims = map(lambda i : get_anchor_iter_dims(i), range(rank))
    stride_dims_list = map(
      lambda num_dims: map(lambda i: anchor_iter_dims[num_dims + i + 1], range(rank - 1 - num_dims)),
      range(rank)
    )
    var_name_and_dims_list = map(
      lambda pair: [pair[0], *pair[1]],
      zip(inputs[0].iter_var_names, stride_dims_list)
    )
    offset_expr = " + ".join(
      map(
        lambda elts: " * ".join(elts),
        var_name_and_dims_list
      )
    )
    assert len(self.output_properties[0].symbolic_shape) == 1, "len(self.output_properties[0]) should be 1"
    return [index_code_gen_value_util.IndexCodeGenValue(
      [f"({offset_expr})"],
      input_iter_dim_splits)
    ]


class CfYieldCodeGen:
  def __init__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    self.index_program_id = index_program_id
    self.op_property = op_property
    self.input_properties = input_properties
    self.output_properties = output_properties
    self.kernel_arg_translator = kernel_arg_translator
    self.anchor_iter_var_names = anchor_iter_var_names
    self.anchor_iter_dim_splits = anchor_iter_dim_splits

  def __call__(self, inputs, mut_kernel_arg_id_registry, mut_lir_code_gen_ctx):
    return []


class OpIndexTranslatorFactory:
  def __init__(self):
    self.op_name2class = OrderedDict([
      ["pd_op.data",                PdOpDataCodeGen],
      ["pd_op.full_int_array",      PdOpFullIntArrayCodeGen],
      ["pd_op.sum",                 PdOpSumCodeGen],
      ["cinn_op.reshape",           CinnOpReshapeCodeGen],
      ["cf.yield",                  CfYieldCodeGen],
    ])

  def __call__(self,
               index_program_id,
               op_property,
               input_properties,
               output_properties,
               kernel_arg_translator,
               anchor_iter_var_names,
               anchor_iter_dim_splits):
    cls = self._get_class(op_property.op_name)
    return cls(
      index_program_id=index_program_id,
      op_property=op_property,
      input_properties=input_properties,
      output_properties=output_properties,
      kernel_arg_translator=kernel_arg_translator,
      anchor_iter_var_names=anchor_iter_var_names,
      anchor_iter_dim_splits=anchor_iter_dim_splits,
    )

  def _get_class(self, op_name):
    return self.op_name2class[op_name]
