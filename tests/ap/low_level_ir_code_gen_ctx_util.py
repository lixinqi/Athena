class CudaLikeIrCodeGenCtx:
    def __init__(self, compute_dtype):
        self.stmts = MutableList()
        self.dtype2type_name = OrderedDict(
            [
                [DataType.float,   "float"],
                [DataType.float16,  "half"],
                [DataType.bfloat16, "__nv_bfloat16"],
                [DataType.int32,     "int"],
                [DataType.int64, "int64_t"],
            ]
        )
        self.compute_dtype = compute_dtype
        self.compute_dtype_name = self.dtype2type_name[self.compute_dtype]
        self.type_cast_str_list = ["", f"static_cast<{self.compute_dtype_name}>"]

    def assign(self, dst, src):
        self.stmts.append(f"{dst.var_name} = {src.var_name};")

    def let(self, var, val_name):
        var_dtype_name = self.dtype2type_name[var.get_dtype()]
        is_same = self.compute_dtype == var.get_dtype()
        type_name = f"{var_dtype_name}" if is_same else f"{self.compute_dtype_name}"
        type_cast_str = "" if is_same else f"static_cast<{self.compute_dtype_name}>"
        self.stmts.append(f"{type_name} {var.var_name} = {type_cast_str}({val_name});")

    def store(self, dtype, dst, offset_var_name, src):
        is_same = dtype == self.dtype2type_name[self.compute_dtype]
        type_cast_str = "" if is_same else f"static_cast<{dtype}>"
        self.stmts.append(f"{dst}[{offset_var_name}] = {type_cast_str}({src});")

    def get_stmts_joined_str(self, indent):
        return f"\n{indent}".join([*self.stmts])
