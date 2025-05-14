
def axpr_dtype_to_nv_dtype_str():
    return OrderedDict([
        [PointerType.const_float_ptr, "const float*"],
        [PointerType.const_float16_ptr, "const half*"],
        [PointerType.const_bfloat16_ptr, "const __nv_bfloat16*"],
        [PointerType.const_int32_t_ptr, "const int*"],
        [PointerType.int32_t_ptr, "int*"],
        [PointerType.float_ptr, "float*"],
        [PointerType.float16_ptr, "half*"],
        [PointerType.bfloat16_ptr, "__nv_bfloat16*"],
        [DataType.float, "float"],
        [DataType.float16, "half"],
        [DataType.bfloat16, "__nv_bfloat16"],
        [DataType.int64_t, "int64_t"],
    ])
