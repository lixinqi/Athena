CUDA_ROOT := /usr/local/cuda
CUTLASS_DIR := /work/abstract_pass/Athena/tests/ap/matmul/cutlass
SOURCE_DIR := /work/abstract_pass/Athena/tests/ap/matmul

# COMPILER OPTIONS
NVCC := nvcc
TARGET := test_main

DEBUG := 0
PROFILE := 0
USE_HALF := 1

LIB_NAME := matmul_kernel
LIB      := lib${LIB_NAME}.so

NVCCFLAGS := -std=c++17 -O3
NVCCFLAGS += -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr
NVCCFLAGS += -I ${CUDA_ROOT}/include -I ${SOURCE_DIR}
NVCCFLAGS += -I ${CUTLASS_DIR}/include -I ${CUTLASS_DIR}/tools/util/include
NVCCFLAGS += -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=${DEBUG}
NVCCFLAGS += -DAP_USE_FLOAT16=${USE_HALF} -DAP_ENABLE_DEBUG=${DEBUG} -DAP_ENABLE_PROFILE=${PROFILE} 

LDFLAGS := -L. -l${LIB_NAME} -lcuda -lcudart

# SRCS = native_kernel.cu matmul_unary_kernel.cu matmul_binary_kernel.cu kernel.cu profile.cu
LIB_SRCS := $(wildcard $(SOURCE_DIR)/*.cu)
LIB_OBJS := $(patsubst %.cu, %.o, $(LIB_SRCS))

TEST_SRCS := $(wildcard $(SOURCE_DIR)/test_*.cc)
TEST_OBJS := $(patsubst %.cc, %.o, ${TEST_SRCS})

# Compile
.PHONY: all clean

all: ${LIB} ${TARGET}

${TARGET}: test_main_matmul_unary.o ${LIB}
	$(NVCC) -o $@ test_main_matmul_unary.o ${LDFLAGS}
	$ echo ">> make success!\n"

${LIB}: ${LIB_OBJS}
	$(NVCC) -shared -o $@ $^

%.o: %.cc
	${NVCC} -c $< -o $@ ${NVCCFLAGS}

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) 

clean: 
	rm -f ${TARGET} ${LIB} ${LIB_OBJS} ${TEST_OBJS}
