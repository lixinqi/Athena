import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util


def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(param_struct_name="args")


class ZipVariadicTemplate:

    def __init__(self, mut_kernel_arg_id_registry):
        self.mut_kernel_arg_id_registry = mut_kernel_arg_id_registry
        self.kernel_arg_translator = make_kernel_arg_translator()
        self.dtype2type_name = OrderedDict(
            [
                [PointerType.const_float_ptr, "const float*"],
                [PointerType.const_float16_ptr, "const half*"],
                [PointerType.const_bfloat16_ptr, "const half*"],
                [PointerType.const_int32_t_ptr, "const int32_t*"],
                [PointerType.float_ptr, "float*"],
                [PointerType.float16_ptr, "half*"],
                [PointerType.bfloat16_ptr, "half*"],
                [PointerType.int32_t_ptr, "int32_t*"],
                [DataType.float, "float"],
                [DataType.float16, "half"],
                [DataType.int64_t, "int64_t"],
            ]
        )
        self.kernel_name = "MoeZipVariadicKernel"
        self.library_name = "moe_zip_variadic_kernel"

    def _register_name(self, pair):
        registry = self.mut_kernel_arg_id_registry
        registry.get_or_create_kernel_arg_id_manul_var_name(
            kernel_arg_id=pair[0], cpp_var_name=pair[1]
        )

    def compile(
        self,
        unzipped_tokens_in_karg,
        zipped_expertwise_rowmap_in_karg,
        expert_routemap_topk_in_karg,
        unzipped_token_probs_in_karg,
        zipped_tokens_out_karg,
        zipped_probs_topk_out_karg,
        topk_kargs,
        num_experts_kargs,
        token_length_kargs,
        total_zipped_tokens_num_kargs,
    ):
        kargs_name_pair_list = [
            [unzipped_tokens_in_karg, "unzipped_tokens_in"],
            [zipped_expertwise_rowmap_in_karg, "zipped_expertwise_rowmap"],
            [expert_routemap_topk_in_karg, "expert_routemap_topk"],
            [unzipped_token_probs_in_karg, "unzipped_token_probs"],
            [zipped_tokens_out_karg, "zipped_tokens_out"],
            [zipped_probs_topk_out_karg, "zipped_probs_topk_out"],
        ]
        map(self._register_name, kargs_name_pair_list)
        project_module = self.make_project()
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=[
                    *self.get_kernel_arg_runtime_getters(),
                    topk_kargs.runtime_getter,
                    num_experts_kargs.runtime_getter,
                    token_length_kargs.runtime_getter,
                    total_zipped_tokens_num_kargs.runtime_getter,
                ]
            ),
        )

    def get_kernel_arg_types(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return map(lambda pair: pair[0].type, all_kernel_arg_id_and_unique_names)

    def get_kernel_arg_runtime_getters(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return map(
            lambda pair: pair[0].runtime_getter, all_kernel_arg_id_and_unique_names
        )

    def get_kernel_arg_list_str(self, for_declare):

        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return f"{type_name} {field_name}" if for_declare else f"{field_name}"

        all_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return ", ".join(
            map(declare_epilogue_arguments_field, all_kernel_arg_id_and_names)
        )

    def make_project(self):
        code_template = """
// auto generated codes
#include "naive_zip.cuh"
namespace ap{

static void RunZipWithVariadicKernel(cudaStream_t cuda_stream_ptr, ${AP_KERNEL_ARGS_DECLARE}, const int64_t topk, const int64_t num_experts, const int64_t token_length, const int64_t total_zipped_tokens_num) {    
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;
  if(topk == 8 && num_experts == 4) {
      zip_naive_kernel<8, 4><<<grid, block, 0, cuda_stream_ptr>>>(${AP_KERNEL_ARGS_CALL}, token_length, total_zipped_tokens_num);
  }
  return;
}

} // namespace ap

extern "C" {
void ${kernel_name}(void* stream_ptr, ${AP_KERNEL_ARGS_DECLARE}, const int64_t topk, const int64_t num_experts, const int64_t token_length, const int64_t total_zipped_tokens_num) {
  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  ap::RunZipWithVariadicKernel(*cuda_stream_ptr, ${AP_KERNEL_ARGS_CALL}, topk, num_experts, token_length, total_zipped_tokens_num);
}
}
  """
        code = (
            code_template.replace(
                "${AP_KERNEL_ARGS_DECLARE}",
                self.get_kernel_arg_list_str(for_declare=True),
            )
            .replace(
                "${AP_KERNEL_ARGS_CALL}",
                self.get_kernel_arg_list_str(for_declare=False),
            )
            .replace("${kernel_name}", self.kernel_name)
        )
        source_dir = "/project/AP/Athena/tests/ap/zip"
        compile_cmd = (
            "nvcc -std=c++17 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        )
        compile_cmd = compile_cmd + " -I " + source_dir
        compile_cmd = compile_cmd + " -DAP_ENABLE_AUTOTUNE=1 -DAP_ENABLE_DEBUG=0"
        compile_cmd = (
            compile_cmd
            + f" --shared {self.library_name}.cu -o lib{self.library_name}.so"
        )
        return CodeModule(
            FuncDeclare(
                DataType.void,
                self.kernel_name,
                [
                    PointerType.void_ptr,
                    *self.get_kernel_arg_types(),
                    DataType.const_int64_t,
                    DataType.const_int64_t,
                    DataType.const_int64_t,
                    DataType.const_int64_t,
                ],
            ),
            Project(
                nested_files=Project.Directory(
                    [f"{self.library_name}.cu", Project.FileContent(code)],
                    ["make.sh", Project.FileContent(compile_cmd)],
                ),
                compile_cmd="sh make.sh",
                so_relative_path=f"lib{self.library_name}.so",
            ),
        )


def KernelDispatch(ctx):
    so_func = ctx.get_so_function("MoeZipVariadicKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
