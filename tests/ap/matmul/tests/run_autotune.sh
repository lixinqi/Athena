#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

#export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/abstract_pass/Athena/tests/ap/matmul:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH

nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o cutlass_matmul"

SWIZZLE_VALUES=("1" "2" "4")

config_id=0
num_configs=26 # 26 for float16, 18 for float
while [ $config_id -le ${num_configs} ]; do
    for value in "${SWIZZLE_VALUES[@]}"; do
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "config_id: ${config_id} swizzle_factor: ${value}"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        cat <<EOF > default_config_id.h
#pragma once

#include "all_tuning_configs.h"

namespace ap {

struct DefaultConfig {
  static constexpr int kConfigId = ${config_id};
  static constexpr int kSwizzleFactor = ${value};
  static constexpr bool kBatched = false;
};

}
EOF
        cat default_config_id.h
        make clean
        make -j4
        ${nsys_args} ./test_main
        echo ""
    done
    config_id=$((config_id + 1))
done
