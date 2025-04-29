#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export NVIDIA_TF32_OVERRIDE=0

sh make_axpr.sh

# AP specific settings
export FLAGS_enable_ap=1
export AP_WORKSPACE_DIR=$(pwd)/ap_workspace
export AP_PATH=$(pwd)/

# CINN related settings
export FLAGS_check_infer_symbolic=1
export FLAGS_enable_pir_api=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_new_group_scheduler=1

export GLOG_vmodule=ap_generic_drr_pass=6

rm -rf ap_workspace/
python $(pwd)/paddle-tests/test_moeunzip.py

