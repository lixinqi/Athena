import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util


def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(param_struct_name="args")


def get_anchor_iter_var_names():
    return ["coord.batch", "coord.row", "coord.column"]


class MoeUnzipVariadicTemplate:
    def __init__(
        self,
        program_translator,
        mut_kernel_arg_id_registry,
    ):
        self.program_translator = program_translator
        self.mut_kernel_arg_id_registry = mut_kernel_arg_id_registry
        self.kernel_arg_translator = make_kernel_arg_translator()
        self.dtype2type_name = OrderedDict(
            [
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
            ]
        )
        self.input_dim_karg_to_shape_access = MutableOrderedDict()
        self.kernel_name = "MoeUnzipVariadicKernel"
        self.library_name = "moe_unzip_variadic_kernel"

    def _register_name(self, pair):
        registry = self.mut_kernel_arg_id_registry
        registry.get_or_create_kernel_arg_id_manul_var_name(
            kernel_arg_id=pair[0], cpp_var_name=pair[1]
        )

    def compile(
        self,
        input0_karg,
        input1_karg,
        input2_karg,
        input3_karg,
        output0_karg,
        output1_karg,
        output2_karg,
        output3_karg,
        tmp_space_karg,
        input0_shape_kargs,
        output_shape_kargs,
    ):
        kargs_name_pair_list = [
            [input0_karg, "input0"],
            [input1_karg, "input1"],
            [input2_karg, "input2"],
            [input3_karg, "input3"],
            [output0_karg, "output"],
            [output1_karg, "output1"],
            [output2_karg, "output2"],
            [output3_karg, "output3"],
            [tmp_space_karg, "tmp_space"],
            *map(
                lambda i: [input0_shape_kargs[i], f"input0_dim{i}"],
                range(len(input0_shape_kargs)),
            ),
            *map(
                lambda i: [output_shape_kargs[i], f"output_dim{i}"],
                range(len(output_shape_kargs)),
            ),
        ]
        print(f"-- kargs_name_pair_list: {kargs_name_pair_list}")
        map(self._register_name, kargs_name_pair_list)

        mut_lir_code_gen_ctx = low_level_ir_code_gen_ctx_util.CudaLikeIrCodeGenCtx(
            compute_dtype=DataType.bfloat16
        )
        self.program_translator.translate(
            mut_kernel_arg_id_registry=self.mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str(indent="    ")
        print("-- moe_unzip_epilogue_code:\n", trivial_code_str)
        project_module = self.make_project(
            trivial_code_str,
            input0_karg,
            input1_karg,
            input2_karg,
            input3_karg,
            output0_karg,
            output1_karg,
            output2_karg,
            output3_karg,
            tmp_space_karg,
            input0_shape_kargs,
            output_shape_kargs,
        )
        return CodeGenResult(
            module=project_module,
            kernel_dispatch_func=KernelDispatch,
            kernel_dispatch_const_data=BuiltinSerializableAttrMap(
                kernel_args_getters=self.get_kernel_arg_runtime_getters()
            ),
        )

    def get_kernel_arg_runtime_getters(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return map(
            lambda pair: pair[0].runtime_getter, all_kernel_arg_id_and_unique_names
        )

    def get_kernel_arg_types(self):
        all_kernel_arg_id_and_unique_names = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name.items()
        )
        return map(lambda pair: pair[0].type, all_kernel_arg_id_and_unique_names)

    def get_kernel_arg_id_var_name(self, kernel_arg_id):
        all_kernel_arg_id2unique_name = (
            self.mut_kernel_arg_id_registry.all_kernel_arg_id2unique_name
        )
        return all_kernel_arg_id2unique_name[kernel_arg_id]

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

    def get_epilogue_arguments_fields_str(self, indent):
        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return f"{type_name} {field_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            map(declare_epilogue_arguments_field, generated_kernel_arg_id_and_names)
        )

    def get_epilogue_arguments_init_str(self, param_obj_name, indent):
        def declare_epilogue_arguments_assign(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            return f"{param_obj_name}.{field_name} = {var_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            map(declare_epilogue_arguments_assign, generated_kernel_arg_id_and_names)
        )

    def get_params_input_shape_init_str(self, input_name, input_shape_kargs, indent):
        def init_input_shape_with_args(i):
            def get_creator():
                return f"{input_name}_shape[{i}]"

            karg_var_name = self.get_kernel_arg_id_var_name(input_shape_kargs[i])
            self.input_dim_karg_to_shape_access.get_or_create(
                karg_var_name, get_creator
            )
            return f"{indent}{input_name}_shape[{i}] = {karg_var_name};"

        shape_vector_init_str = (
            f"{input_name}_shape.resize({len(input_shape_kargs)});\n"
        )
        return shape_vector_init_str + "\n".join(
            map(init_input_shape_with_args, range(len(input_shape_kargs)))
        )

    def make_project(
        self,
        trivial_code_str,
        input0_karg,
        input1_karg,
        input2_karg,
        input3_karg,
        output0_karg,
        output1_karg,
        output2_karg,
        output3_karg,
        tmp_space_karg,
        input0_shape_kargs,
        output_shape_kargs,
    ):
        code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_bf16.h>
#include <vector>
#include "moe_unzip.cuh"
#include <iostream>

namespace ap{

template <typename T>
struct MoeUnzipEpilogueFunctor {
  struct Arguments {
    ${AP_EPILOGUE_ARGUMENTS_FIELDS}
  };

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x) const {
    T out;
    ${AP_EPILOGUE_COMPUTATION_STATEMENTS}
    return out;
  }
};

}// namespace ap

extern "C" {

void ${kernel_name}(void* stream_ptr, ${AP_KERNEL_ARGS_DECLARE}) {
  using ElementT = ${output_dtype};
  using ElementComputeT = float;

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  std::cout << "start tokens_unzip_stable_kernel" << std::endl;
  ap::tokens_unzip_stable<ElementT, ElementT, ap::MoeUnzipEpilogueFunctor>(*cuda_stream_ptr, ${input0}, ${input1}, ${input2}, ${input3}, 384, 8, 4, ${output0}, ${output1}, ${output2}, ${output3}, (int*)${tmp_space}, ${rows}, ${output_rows}, ${cols});
   
}
}
  """

        output_dtype = self.dtype2type_name[output0_karg.type.data_type]
        code = (
            code_template
            .replace(
                "${AP_KERNEL_ARGS_DECLARE}",
                self.get_kernel_arg_list_str(for_declare=True),
            )
            .replace(
                "${AP_EPILOGUE_ARGUMENTS_FIELDS}",
                self.get_epilogue_arguments_fields_str(indent="    "),
            )
            .replace(
                "${AP_EPILOGUE_COMPUTATION_STATEMENTS}", trivial_code_str
            )
            .replace("${kernel_name}", self.kernel_name)
            .replace("${input0}", self.get_kernel_arg_id_var_name(input0_karg))
            .replace("${input1}", self.get_kernel_arg_id_var_name(input1_karg))
            .replace("${input2}", self.get_kernel_arg_id_var_name(input2_karg))
            .replace("${input3}", self.get_kernel_arg_id_var_name(input3_karg))
            .replace("${output0}", self.get_kernel_arg_id_var_name(output0_karg))
            .replace("${output1}", self.get_kernel_arg_id_var_name(output1_karg))
            .replace("${output2}", self.get_kernel_arg_id_var_name(output2_karg))
            .replace("${output3}", self.get_kernel_arg_id_var_name(output3_karg))
            .replace("${tmp_space}", self.get_kernel_arg_id_var_name(tmp_space_karg))
            .replace("${rows}", f"{input0_shape_kargs[0].value}")
            .replace("${output_rows}", f"{output_shape_kargs[0].value}")
            .replace("${cols}", f"{input0_shape_kargs[1].value}")
            .replace("${output_dtype}", output_dtype)
        )

        source_dir = "/workspace/Athena/tests/ap/moe_utils"
        compile_cmd = (
            "nvcc -std=c++17 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        )
        compile_cmd = compile_cmd + " -I " + source_dir
        compile_cmd = (
            compile_cmd
            + f" --shared {self.library_name}.cu -o lib{self.library_name}.so"
        )

        return CodeModule(
            FuncDeclare(
                DataType.void,
                self.kernel_name,
                [PointerType.void_ptr, *self.get_kernel_arg_types()],
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
    so_func = ctx.get_so_function("MoeUnzipVariadicKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
