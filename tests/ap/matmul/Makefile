CUDA_ROOT   := /usr/local/cuda
CUTLASS_DIR := /work/abstract_pass/Athena/tests/ap/matmul/cutlass
SOURCE_DIR  := /work/abstract_pass/Athena/tests/ap/matmul
BUILD_DIR   := ${SOURCE_DIR}/build

DEBUG     := 0
PROFILE   := 0
USE_HALF  := 1

LIB_NAME  := matmul_kernel
LIB       := ${BUILD_DIR}/lib${LIB_NAME}.so
TARGET    := ${BUILD_DIR}/test_main

# COMPILER OPTIONS
NVCC      := nvcc

NVCCFLAGS := -std=c++17 -O3
NVCCFLAGS += -Xcompiler=-fPIC -arch=sm_80 --expt-relaxed-constexpr
NVCCFLAGS += -I ${CUDA_ROOT}/include -I ${SOURCE_DIR}
NVCCFLAGS += -I ${CUTLASS_DIR}/include -I ${CUTLASS_DIR}/tools/util/include
NVCCFLAGS += -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=${DEBUG}
NVCCFLAGS += -DAP_USE_FLOAT16=${USE_HALF} -DAP_ENABLE_DEBUG=${DEBUG} -DAP_ENABLE_PROFILE=${PROFILE} 

LDFLAGS   := -L./build -l${LIB_NAME} -lcuda -lcudart

# SRCS = native_kernel.cu matmul_unary_kernel.cu matmul_binary_kernel.cu kernel.cu profile.cu
LIB_SRCS  := $(wildcard ${SOURCE_DIR}/*.cu)
LIB_OBJS  := $(patsubst ${SOURCE_DIR}/%.cu, ${BUILD_DIR}/%.o, ${LIB_SRCS})

TEST_SRCS := $(wildcard ${SOURCE_DIR}/test_*.cc)
TEST_OBJS := $(patsubst ${SOURCE_DIR}/%.cc, ${BUILD_DIR}/%.o, ${TEST_SRCS})

# Compile
.PHONY: all clean

all: ${LIB} ${TARGET}

${BUILD_DIR}:
	mkdir -p ${BUILD_DIR}

${TARGET}: ${BUILD_DIR}/test_main_matmul_unary.o ${LIB}
	$(NVCC) -o $@ ${BUILD_DIR}/test_main_matmul_unary.o ${LDFLAGS}
	$ echo ">> make success!"

${LIB}: ${LIB_OBJS}
	$(NVCC) -shared -o $@ $^

${BUILD_DIR}/%.o: %.cc | ${BUILD_DIR}
	${NVCC} -c $< -o $@ ${NVCCFLAGS}

${BUILD_DIR}/%.o: %.cu | ${BUILD_DIR}
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) 

clean: 
	rm -f ${TARGET} ${LIB} ${LIB_OBJS} ${TEST_OBJS}
