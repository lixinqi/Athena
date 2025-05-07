#!/bin/bash

FILENAMES_ARRAY=(
    "index_code_gen_value_util"
    "index_drr_pass_util"
    "kernel_arg_translator_util"
    "index_program_translator_util"
    "low_level_ir_code_gen_ctx_util"
    "kernel_arg_id_util"
    "code_gen_value_util"
    "op_index_translator_util"
    "op_compute_translator_util"
    "program_translator_util"
    "__main__"
    "topo_drr_pass"
    "op_convertion_drr_pass"
    "umprime"
    "access_topo_drr"
    "abstract_drr"
    "ap_tpl_codegen"
    "matmul_variadic_tpl"
    "matmul_epilogue_pass"
    "test_matmul_binary"
    "test_matmul_epilogue"
    "unzip_tpl"
    "test_moeunzip"
)
for filename in "${FILENAMES_ARRAY[@]}"
do
    echo "-- Convert ${filename}.py -> ${filename}.py.json"
    python ../../athena/advanced_pass/py_to_json.py ${filename}.py ${filename}.py.json
done
