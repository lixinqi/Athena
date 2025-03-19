#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES="4"
export NVIDIA_TF32_OVERRIDE=0
sh make_axpr.sh test_matmul_epilogue
FILE_NUM=$1
LOG_DIR=$2
TEST_DIR=/work/PaddleTest/framework/e2e/PaddleLT_new/layerE2Ecase/matmul-related-subgraphs/
FILENAME=${TEST_DIR}/${FILE_NUM}.py

export FLAGS_enable_ap=1
export NVIDIA_TF32_OVERRIDE=1
export AP_WORKSPACE_DIR=$(pwd)/ap_workspace
export AP_PATH=$(pwd)/
export ATHENA_ENABLE_TRY_RUN=0
export FLAGS_check_infer_symbolic=1
export FLAGS_enable_pir_api=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_prim_enable_dynamic=True
export FLAGS_prim_all=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_new_group_scheduler=1
# export FLAGS_cinn_enable_vectorize=true

# export GLOG_v=6

# export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${FILENAME}"

# ${nsys_args}
GLOG_vmodule=ap_lower_fusion_op_pass=8 python $FILENAME 2>&1 | tee "${LOG_DIR}/log_${FILE_NUM}.txt"
