PYTORCH_LIB_DIR := /home/fy/pytorch/torch/lib


PYTHON := python3
NVCC_COMPILE := nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
BUILD_DIR := dense
INCLUDE_DIRS := TH THC THCUNN include include/TH
TORCH_FFI_BUILD := build.py
BN_KERNEL := $(BUILD_DIR)/batchnormp_kernel.so
TORCH_FFI_TARGET := $(BUILD_DIR)/batch_norm/_batch_norm.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$(PYTORCH_LIB_DIR)/$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(BN_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/batchnormp_kernel.so: src/batchnormp_cuda_kernel.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS) -Isrc

clean:
	$(RM_RF) $(BUILD_DIR)