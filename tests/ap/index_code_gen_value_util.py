class IndexCodeGenValue:
  def __init__(self, iter_var_names, iter_dim_splits):
    self.iter_var_names = iter_var_names
    self.iter_dim_splits = iter_dim_splits
    self.const_data = None
