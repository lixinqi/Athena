import low_level_ir_code_gen_ctx_util
import kernel_arg_translator_util


def make_kernel_arg_translator():
    return kernel_arg_translator_util.KernelArgTranslator(param_struct_name="args")


def get_anchor_iter_var_names():
    return ["coord.batch", "coord.row", "coord.column"]


class MatmulBinaryTemplate:
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
        trivial_code_str = mut_lir_code_gen_ctx.get_stmts_joined_str()
        print("-- matmul_binary_epilogue_code:\n", trivial_code_str)
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

    def get_epilogue_arguments_fields_str(self):
        def declare_epilogue_arguments_field(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            dtype = kernel_arg_id.type
            type_name = self.dtype2type_name[dtype]
            return f"    {type_name} {field_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return "\n".join(
            map(declare_epilogue_arguments_field, generated_kernel_arg_id_and_names)
        )

    def get_epilogue_arguments_init_str(self, param_obj_name):
        def declare_epilogue_arguments_assign(pair):
            kernel_arg_id = pair[0]
            var_name = pair[1]
            field_name = self.kernel_arg_translator.get_param_struct_field_name(
                var_name
            )
            return f"  {param_obj_name}.{field_name} = {var_name};"

        generated_kernel_arg_id_and_names = (
            self.mut_kernel_arg_id_registry.generated_kernel_arg_id2unique_name.items()
        )
        return "\n".join(
            map(declare_epilogue_arguments_assign, generated_kernel_arg_id_and_names)
        )

    def get_input_shape_init_str(self, input_name, input_shape_kargs, indent):
        def init_input_shape_with_args(i):
            karg = input_shape_kargs[i]
            return f"{indent}{input_name}_shape[{i}] = {self.get_kernel_arg_id_var_name(karg)};"

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

}

extern "C" {

void MatmulBinaryKernel(void* stream_ptr, AP_KERNEL_ARGS_DECLARE) {
  std::vector<int64_t> $input0_shape;
  AP_INPUT0_SHAPE_INIT

  std::vector<int64_t> $input1_shape;
  AP_INPUT1_SHAPE_INIT

  cudaStream_t* cuda_stream_ptr = reinterpret_cast<cudaStream_t*>(stream_ptr);
  ap::GemmEpilogueParams params(
      *cuda_stream_ptr, $input0, $input1, nullptr, $output, $input0_shape, $input1_shape);

  using ElementT = AP_GENERATED_ELEMENT_DTYPE;
  using ElementComputeT = float;

  typename ap::VariadicEpilogueFunctor<ElementComputeT>::Arguments epilogue_args;

AP_EPILOGUE_ARGUMENTS_INIT

  ap::CutlassMatmulAddVariadic<ElementT, ElementComputeT, ap::VariadicEpilogueFunctor>(params, epilogue_args);
}
}

  """

        output_dtype = self.dtype2type_name[output_karg.type.data_type]
        code = (
            code_template.replace(
                "AP_GENERATED_BINARY_EPILOGUE_STRING", trivial_code_str
            )
            .replace("AP_GENERATED_ELEMENT_DTYPE", output_dtype)
            .replace("AP_KERNEL_ARGS_DECLARE", self.get_kernel_arg_list_str())
            .replace(
                "AP_INPUT0_SHAPE_INIT",
                self.get_input_shape_init_str("$input0", input0_shape_kargs, "  "),
            )
            .replace(
                "AP_INPUT1_SHAPE_INIT",
                self.get_input_shape_init_str("$input1", input1_shape_kargs, "  "),
            )
            .replace(
                "AP_EPILOGUE_ARGUMENTS_FIELDS", self.get_epilogue_arguments_fields_str()
            )
            .replace(
                "AP_EPILOGUE_ARGUMENTS_INIT",
                self.get_epilogue_arguments_init_str("epilogue_args"),
            )
            .replace("$input0", self.get_kernel_arg_id_var_name(input0_karg))
            .replace("$input1", self.get_kernel_arg_id_var_name(input1_karg))
            .replace("$output", self.get_kernel_arg_id_var_name(output_karg))
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
            + " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 "
        )
        compile_cmd = (
            compile_cmd
            + " --shared matmul_binary_kernel.cu -o libmatmul_binary_kernel.so"
        )

        return CodeModule(
            FuncDeclare(
                DataType.void,
                "MatmulBinaryKernel",
                [PointerType.void_ptr, *self.get_kernel_arg_types()],
            ),
            Project(
                nested_files=Project.Directory(
                    ["matmul_binary_kernel.cu", Project.FileContent(code)],
                    ["make.sh", Project.FileContent(compile_cmd)],
                ),
                compile_cmd="sh make.sh",
                so_relative_path="libmatmul_binary_kernel.so",
            ),
        )


def KernelDispatch(ctx):
    so_func = ctx.get_so_function("MatmulBinaryKernel")
    stream_ptr = ctx.device_ctx.get_stream_addr_as_void_ptr()
    getters = ctx.kernel_dispatch_const_data.kernel_args_getters
    args = [stream_ptr, *map(lambda getter: getter(ctx), getters)]
    apply(so_func, args)
