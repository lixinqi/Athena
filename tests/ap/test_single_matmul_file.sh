#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES="5"
export NVIDIA_TF32_OVERRIDE=0
sh make_axpr.sh test_matmul_epilogue
FILE_NUM=$1
LOG_DIR=$2
TEST_DIR=/work/PaddleTest/all_case_v2/
FILENAME=${TEST_DIR}/${FILE_NUM}
export FLAGS_test_accuracy=0
export FLAGS_check_infer_symbolic=1
export FLAGS_prim_enable_dynamic=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_enable_pir_api=1
export AP_WORKSPACE_DIR=$(pwd)/ap_workspace
export AP_PATH=$(pwd)/
export ATHENA_ENABLE_TRY_RUN=0
export FLAGS_cinn_bucket_compile=True
export FLAGS_cinn_new_group_scheduler=1
nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi\
    --force-overwrite true -o ${LOG_DIR}/${FILE_NUM}"
# export FLAGS_cinn_enable_vectorize=true
# export GLOG_v=6
# export GLOG_vmodule=ap_generic_drr_pass=6

FLAGS_enable_ap=0 FLAGS_ap_performance=0 FLAGS_prim_all=False ${nsys_args} timeout 20 python $FILENAME 2>&1 | tee "${LOG_DIR}/log_${FILE_NUM}.txt" 
python parse_nsys_stats.py "${LOG_DIR}/${FILE_NUM}.sqlite" "pd" > "${LOG_DIR}/time_${FILE_NUM}_pd.txt" 
sleep 2

FLAGS_enable_ap=0 FLAGS_ap_performance=1 FLAGS_prim_all=True ${nsys_args} timeout 30 python $FILENAME 2>&1 | tee -a "${LOG_DIR}/log_${FILE_NUM}.txt" 
python parse_nsys_stats.py "${LOG_DIR}/${FILE_NUM}.sqlite" "cinn" > "${LOG_DIR}/time_${FILE_NUM}_cinn.txt" 
sleep 2

FLAGS_enable_ap=1 FLAGS_ap_performance=1 FLAGS_prim_all=True ${nsys_args} timeout 210 python $FILENAME 2>&1 | tee  -a "${LOG_DIR}/log_${FILE_NUM}.txt"
python parse_nsys_stats.py "${LOG_DIR}/${FILE_NUM}.sqlite" "ap" > "${LOG_DIR}/time_${FILE_NUM}_ap.txt" 
sleep 2
