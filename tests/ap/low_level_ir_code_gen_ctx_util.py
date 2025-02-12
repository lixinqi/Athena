class CudaLikeIrCodeGenCtx:
    def __init__(self, compute_dtype):
        self.stmts = MutableList()
        self.dtype2type_name = OrderedDict(
            [
                [DataType.float, "float"],
                [DataType.float16, "half"],
                [DataType.int32, "int"],
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
        type_name_list = [f"{var_dtype_name}", f"{self.compute_dtype_name}"]
        index = int(self.compute_dtype != var.get_dtype())
        type_name = type_name_list[index]
        type_cast_str = self.type_cast_str_list[index]
        self.stmts.append(f"{type_name} {var.var_name} = {type_cast_str}({val_name});")

    def get_stmts_joined_str(self):
        return "\n".join([*self.stmts])
