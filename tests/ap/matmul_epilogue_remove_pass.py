import access_topo_drr
import pir

class RemoveDataOpPairPass(access_topo_drr.DrrPass):
  def __init__(self, src_data_op_name, dst_data_op_name):
    self.src_data_op_name = pir.a_str(src_data_op_name)
    self.dst_data_op_name = pir.a_str(dst_data_op_name)
  def source_pattern(self, o, t):
    o.src_data_op = o.ap_native_op("pd_op.data")
    o.src_data_op(
      [],
      [t.input0]
    )
    o.dst_data_op = o.ap_native_op("pd_op.data")
    o.dst_data_op(
      [],
      [t.input1]
    )
    o.up_spider_op = o.ap_native_op("ap_op.up_spider")
    o.up_spider_op(
      [t.input0, t.input1],
      []
    )
  def constraint(self, o, t):
    return [o.src_data_op.name, o.dst_data_op.name] == [self.src_data_op_name, self.dst_data_op_name]
  def result_pattern(self, o, t):
    pass

class RemoveDataOp2SumOp2DataOpPass(access_topo_drr.DrrPass):
  def __init__(self, src_data_op_name, dst_data_op_name):
    self.src_data_op_name = pir.a_str(src_data_op_name)
    self.dst_data_op_name = pir.a_str(dst_data_op_name)

  def source_pattern(self, o, t):
    o.src_data_op = o.ap_native_op("pd_op.data")
    o.src_data_op.name = self.src_data_op_name
    o.src_data_op(
      [],
      [t.input0]
    )
    o.full_int_array_op = o.ap_native_op("pd_op.full_int_array")
    o.full_int_array_op(
      [],
      [t.axis]
    )
    o.sum_op = o.ap_native_op("pd_op.sum")
    o.sum_op(
      [t.input0, t.axis],
      [t.sum_out]
    )
    o.dst_data_op = o.ap_native_op("pd_op.data")
    o.dst_data_op.name = self.dst_data_op_name
    o.dst_data_op(
      [],
      [t.input1]
    )
    o.up_spider_op = o.ap_native_op("ap_op.up_spider")
    o.up_spider_op(
      [t.sum_out, t.input1],
      []
    )

  def result_pattern(self, o, t):
    pass

class RemoveElementInputIndexPass(access_topo_drr.DrrPass):

  def __init__(self, src_data_op_name, dst_load_from_global_op_name):
    self.src_data_op_name = pir.a_str(src_data_op_name)
    self.dst_load_from_global_op_name = pir.a_str(dst_load_from_global_op_name)

  def source_pattern(self, o, t):
    o.src_data_op = o.ap_native_op("pd_op.data")
    o.src_data_op.name = self.src_data_op_name
    o.src_data_op(
      [],
      [t.src_input]
    )

    o.dst_load_from_global_op = o.ap_native_op("ap_op.load_from_global")
    o.dst_load_from_global_op.index_func_unique_id = self.dst_load_from_global_op_name
    o.dst_load_from_global_op(
      [t.dst_input],
      [t.dst_load_from_global_output]
    )
    o.up_spider_op = o.ap_native_op("ap_op.up_spider")
    o.up_spider_op(
      [t.src_input, t.dst_load_from_global_output],
      []
    )

  def result_pattern(self, o, t):
    pass

class RemoveBroadcastInputIndexPass(access_topo_drr.DrrPass):
  def __init__(self, src_data_op_name, dst_load_from_global_op_name):
    self.src_data_op_name = pir.a_str(src_data_op_name)
    self.dst_load_from_global_op_name = pir.a_str(dst_load_from_global_op_name)

  def source_pattern(self, o, t):
    o.src_data_op = o.ap_native_op("pd_op.data")
    o.src_data_op.name = self.src_data_op_name
    o.src_data_op(
      [],
      [t.input0]
    )
    o.full_int_array_op = o.ap_native_op("pd_op.full_int_array")
    o.full_int_array_op(
      [],
      [t.axis]
    )
    o.sum_op = o.ap_native_op("pd_op.sum")
    o.sum_op(
      [t.input0, t.axis],
      [t.sum_out]
    )
    o.dst_load_from_global_op = o.ap_native_op("ap_op.load_from_global")
    o.dst_load_from_global_op.index_func_unique_id = self.dst_load_from_global_op_name
    o.dst_load_from_global_op(
      [t.dst_input],
      [t.dst_load_from_global_output]
    )
    o.up_spider_op = o.ap_native_op("ap_op.up_spider")
    o.up_spider_op(
      [t.sum_out, t.dst_load_from_global_output],
      []
    )

  def result_pattern(self, o, t):
    pass

class RemoveOutputIndexPass(access_topo_drr.DrrPass):

  def __init__(self, src_data_op_name, dst_store_to_global_op_name):
    self.src_data_op_name = pir.a_str(src_data_op_name)
    self.dst_store_to_global_op_name = pir.a_str(dst_store_to_global_op_name)

  def source_pattern(self, o, t):
    o.src_data_op = o.ap_native_op("pd_op.data")
    o.src_data_op.name = self.src_data_op_name
    o.src_data_op(
      [],
      [t.src_input]
    )
    o.down_spider_op = o.ap_native_op("ap_op.down_spider")
    o.down_spider_op(
      [t.src_input],
      [t.dst_output_val]
    )
    o.dst_store_to_global_op = o.ap_native_op("ap_op.store_to_global")
    o.dst_store_to_global_op.index_func_unique_id = self.dst_store_to_global_op_name
    o.dst_store_to_global_op(
      [t.dst_output, t.dst_output_val],
      []
    )
    # o.up_spider_op = o.ap_native_op("ap_op.up_spider")
    # o.up_spider_op(
    #   [t.src_input, t.dst_output_val],
    #   []
    # )

  def result_pattern(self, o, t):
    pass
