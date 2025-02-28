import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util


def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(param_struct_name="args")


def get_anchor_iter_var_names():
    return ["coord.batch", "coord.row", "coord.column"]


def is_in_tensor_karg(kernel_arg_id):
    kernel_arg_id_type_name = f"{type(kernel_arg_id)}".replace("<class '", "").replace("'>", "")
    return kernel_arg_id_type_name == "InTensorDataPtrKernelArgId"


class MatmulVariadicTemplate:
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
                [PointerType.float_ptr, "float*"],
                [PointerType.float16_ptr, "half*"],
                [DataType.float, "float"],
                [DataType.float16, "half"],
                [DataType.int64_t, "int64_t"],
            ]
        )
        self.input_dim_karg_to_shape_access = MutableOrderedDict()
        self.input_tensor_karg_to_shape_access = MutableOrderedDict()
        self.kernel_name = "MatmulVariadicKernel"
        self.library_name = "matmul_variadic_kernel"

    def _register_name(self, pair):
        registry = self.mut_kernel_arg_id_registry
        registry.get_or_create_kernel_arg_id_manul_var_name(
            kernel_arg_id=pair[0], cpp_var_name=pair[1]
        )

    def compile(
        self,
        input0_karg,
        input1_karg,
        output_karg,
        input0_shape_kargs,
        input1_shape_kargs,
    ):
        kargs_name_pair_list = [
            [input0_karg, "input0"],
            [input1_karg, "input1"],
            [output_karg, "output"],
            *map(
                lambda i: [input0_shape_kargs[i], f"input0_dim{i}"],
                range(len(input0_shape_kargs)),
            ),
            *map(
                lambda i: [input1_shape_kargs[i], f"input1_dim{i}"],
                range(len(input1_shape_kargs)),
            ),
        ]
        print(f"-- kargs_name_pair_list: {kargs_name_pair_list}")
        map(self._register_name, kargs_name_pair_list)

        mut_lir_code_gen_ctx = low_level_ir_code_gen_ctx_util.CudaLikeIrCodeGenCtx(
            compute_dtype=DataType.float
        )
        self.program_translator.translate(
            mut_kernel_arg_id_registry=self.mut_kernel_arg_id_registry,
            mut_lir_code_gen_ctx=mut_lir_code_gen_ctx,
        )
        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str(indent="    ")
        print("-- matmul_variadic_epilogue_code:\n", trivial_code_str)
        project_module = self.make_project(
            trivial_code_str,
            input0_karg,
            input1_karg,
            output_karg,
            input0_shape_kargs,
            input1_shape_kargs,
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

    def get_kernel_arg_list_str(self):
        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return f"{type_name} {field_name}"

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

    def get_epilogue_arguments_init_str(self, obj_name, params_name, output_dtype, indent):
        def declare_epilogue_arguments_assign(pair):
            kernel_arg_id = pair[0]
            is_in_tensor_type = is_in_tensor_karg(kernel_arg_id)

            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            def get_in_tensor_statement():
                param_name_for_var = self.input_tensor_karg_to_shape_access[var_name]
                return f"reinterpret_cast<const {output_dtype} *>({params_name}.{param_name_for_var})"
            def get_dim_expr_statement():
                param_name_for_var = self.input_dim_karg_to_shape_access[var_name]
                return f"{params_name}.{param_name_for_var}"
            statement = get_in_tensor_statement() if is_in_tensor_type else get_dim_expr_statement()
            return f"{obj_name}.{field_name} = {statement};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            map(declare_epilogue_arguments_assign, generated_kernel_arg_id_and_names)
        )

    def get_params_epilogue_ptrs_init_str(self, obj_name, indent):
        in_tensor_id = 0
        def declare_params_epilogue_arguments_assign(pair):
            def get_creator():
                return f"{obj_name}[{in_tensor_id}]"

            kernel_arg_id = pair[0]
            is_in_tensor_type = is_in_tensor_karg(kernel_arg_id)
            def generate_statement():
                self.input_tensor_karg_to_shape_access.get_or_create(pair[1], get_creator)
                statement = f"{obj_name}.push_back({pair[1]});"
                in_tensor_id = in_tensor_id + 1
                return statement
            return generate_statement() if is_in_tensor_type else ""

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return f"\n{indent}".join(
            map(declare_params_epilogue_arguments_assign, generated_kernel_arg_id_and_names)
        )

    def get_params_input_shape_init_str(self, input_name, input_shape_kargs, indent):
        def init_input_shape_with_args(i):
            def get_creator():
                return f"{input_name}_shape[{i}]"
            karg_var_name = self.get_kernel_arg_id_var_name(input_shape_kargs[i])
            self.input_dim_karg_to_shape_access.get_or_create(karg_var_name, get_creator)
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
        output_karg,
        input0_shape_kargs,
        input1_shape_kargs,
    ):
        code_template = """
// auto generated codes
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

#include "cutlass_matmul.cuh"
#include "profile.h"

namespace ap {

template <typename T>
struct VariadicEpilogueFunctor {
  struct Arguments {
    AP_EPILOGUE_ARGUMENTS_FIELDS
  };

  // Note: need to support vectorized operation
  __forceinline__ __host__ __device__
  T operator()(T x, const Arguments& args, const MatrixCoord& coord) const {
    T out;
    AP_GENERATED_BINARY_EPILOGUE_STRING
    return out;
  }
};

template <int TuningConfigId>
static void RunMatmulWithVariadicKernel(const GemmEpilogueParams &params) {
  using ElementT = ${output_dtype};
  using ElementComputeT = float;

  typename ap::VariadicEpilogueFunctor<ElementComputeT>::Arguments epilogue_args;

  AP_EPILOGUE_ARGUMENTS_INIT

  ap::CutlassMatmulAddVariadic<ElementT, ElementComputeT, ap::VariadicEpilogueFunctor, TuningConfigId>(params, epilogue_args);
}

} // namespace ap

extern "C" {

void ${kernel_name}(void* stream_ptr, AP_KERNEL_ARGS_DECLARE) {
  std::vector<int64_t> ${input0}_shape;
  AP_PARAMS_INPUT0_SHAPE_INIT

  std::vector<int64_t> ${input1}_shape;
  AP_PARAMS_INPUT1_SHAPE_INIT

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  ap::GemmEpilogueParams params(
      *cuda_stream_ptr, ${input0}, ${input1}, nullptr, ${output}, ${input0}_shape, ${input1}_shape, std::vector<int64_t>{});

  std::vector<const void *> epilogue_in_ptrs;
  AP_PARAMS_EPILOGUE_PTRS_INIT

  params.SetEpilogues(epilogue_in_ptrs);

#if AP_ENABLE_AUTOTUNE
  AP_AUTOTUNE_${output_dtype}(ap::RunMatmulWithVariadicKernel);
#else
  ap::RunMatmulWithVariadicKernel<ap::DefaultConfig::kConfigId>(params);
#endif
}
}
  """

        output_dtype = self.dtype2type_name[output_karg.type.data_type]
        code = (
            code_template.replace(
                "AP_GENERATED_BINARY_EPILOGUE_STRING", trivial_code_str
            )
            .replace("AP_KERNEL_ARGS_DECLARE", self.get_kernel_arg_list_str())
            .replace(
                "AP_PARAMS_INPUT0_SHAPE_INIT",
                self.get_params_input_shape_init_str("${input0}", input0_shape_kargs, indent="  "),
            )
            .replace(
                "AP_PARAMS_INPUT1_SHAPE_INIT",
                self.get_params_input_shape_init_str("${input1}", input1_shape_kargs, indent="  "),
            )
            .replace(
                "AP_PARAMS_EPILOGUE_PTRS_INIT",
                self.get_params_epilogue_ptrs_init_str("epilogue_in_ptrs", indent="  "),
            )
            .replace(
                "AP_EPILOGUE_ARGUMENTS_FIELDS", self.get_epilogue_arguments_fields_str(indent="    ")
            )
            .replace(
                "AP_EPILOGUE_ARGUMENTS_INIT",
                self.get_epilogue_arguments_init_str("epilogue_args", "params", output_dtype, indent="  "),
            )
            .replace("${kernel_name}", self.kernel_name)
            .replace("${input0}", self.get_kernel_arg_id_var_name(input0_karg))
            .replace("${input1}", self.get_kernel_arg_id_var_name(input1_karg))
            .replace("${output}", self.get_kernel_arg_id_var_name(output_karg))
            .replace("${output_dtype}", output_dtype)
        )

        source_dir = "/work/abstract_pass/Athena/tests/ap/matmul"
        cutlass_dir = "/work/abstract_pass/Athena/tests/ap/matmul/cutlass"
        compile_cmd = (
            "nvcc -std=c++17 -O3 -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr"
        )
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/include"
        compile_cmd = compile_cmd + " -I " + cutlass_dir + "/tools/util/include"
        compile_cmd = compile_cmd + " -I " + source_dir
        compile_cmd = (
            compile_cmd
            + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0"
        )
        compile_cmd = compile_cmd + " -DAP_ENABLE_AUTOTUNE=1 -DAP_ENABLE_DEBUG=0"
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
    so_func = ctx.get_so_function("MatmulVariadicKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
