class QuantHorizontalTemplate:
    def compile(
        self,
        input0_karg,
        input1_karg,
        output0_karg,
        scale0_karg,
        output1_karg,
        scale1_karg,
    ):
        project_module = self.make_project(
            input0_karg,
            input1_karg,
            output0_karg,
            scale0_karg,
            output1_karg,
            scale1_karg,
        )
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=[
                    input0_karg.runtime_getter,
                    input1_karg.runtime_getter,
                    output0_karg.runtime_getter,
                    scale0_karg.runtime_getter,
                    output1_karg.runtime_getter,
                    scale1_karg.runtime_getter,
                ]
            ),
        )

    def make_project(
        self,
        input0_karg,
        input1_karg,
        output0_karg,
        scale0_karg,
        output1_karg,
        scale1_karg,
    ):
        code = """
extern "C" {
void DualQuantKernel(void* stream_ptr, const float* input0, const float* input1, float* output0, float*scale0, float*outpu1, float*scale1) {
}
}
"""
        compile_cmd = "nvcc --compiler-options '-fPIC' --shared dual_quant_kernel.cu -o libdual_quant_kernel.so"

        return CodeModule(
            FuncDeclare(
                DataType.void,
                "DualQuantKernel",
                [
                    PointerType.void_ptr,
                    PointerType.const_float_ptr,
                    PointerType.const_float_ptr,
                    PointerType.float_ptr,
                    PointerType.float_ptr,
                    PointerType.float_ptr,
                    PointerType.float_ptr,
                ],
            ),
            Project(
                nested_files=Project.Directory(
                    ["dual_quant_kernel.cu", Project.FileContent(code)],
                    ["make.sh", Project.FileContent(compile_cmd)],
                ),
                compile_cmd="sh make.sh",
                so_relative_path="libdual_quant_kernel.so",
            ),
        )


def KernelDispatch(ctx):
    so_func = ctx.get_so_function("DualQuantKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
