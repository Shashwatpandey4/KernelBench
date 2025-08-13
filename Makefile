SHELL := /bin/bash

NVCC := /usr/local/cuda-12.8/bin/nvcc
GPU_ARCH ?= sm_86
HOST_CXX ?= /usr/bin/g++-12
NVCCFLAGS := -O3 --use_fast_math -std=c++14 -arch=$(GPU_ARCH) -ccbin=$(HOST_CXX) \
             -cudart shared -Xlinker -rpath,/usr/local/cuda-12.8/lib64

MOJOENV := .mojoenv
MOJOFLAGS := -O3
MOJO_FAST_FLAGS := -O1  # Faster compilation for development

KERNEL ?= hello

.PHONY: run_triton run_cuda run_mojo run_mojo_aot run_mojo_fast build_mojo build_mojo_fast clean

# CUDA build rule
build/cuda/%: kernels/cuda/%.cu
	@mkdir -p build/cuda
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Mojo AOT build rule
build/mojo/%: kernels/mojo/%.mojo
	@mkdir -p build/mojo
	@cd $(MOJOENV) && pixi run mojo build $(MOJOFLAGS) ../kernels/mojo/$(notdir $<) -o ../build/mojo/$(notdir $*)

# Mojo fast build rule (O1 optimization)
build/mojo/%-fast: kernels/mojo/%.mojo
	@mkdir -p build/mojo
	@cd $(MOJOENV) && pixi run mojo build $(MOJO_FAST_FLAGS) ../kernels/mojo/$(notdir $<) -o ../build/mojo/$(notdir $*)-fast

# CUDA execution
run_cuda: build/cuda/$(KERNEL)
	@./build/cuda/$(KERNEL)

# Mojo JIT execution (original behavior)
run_mojo:
	@cd $(MOJOENV) && pixi run mojo run ../kernels/mojo/$(KERNEL).mojo

# Mojo AOT execution (new option)
run_mojo_aot: build/mojo/$(KERNEL)
	@./build/mojo/$(KERNEL)

# Mojo fast AOT execution (O1 - faster compilation)
run_mojo_fast: build/mojo/$(KERNEL)-fast
	@./build/mojo/$(KERNEL)-fast

# Build Mojo binary without running
build_mojo: build/mojo/$(KERNEL)

# Build Mojo binary with fast compilation
build_mojo_fast: build/mojo/$(KERNEL)-fast

# Triton execution
run_triton:
	@. .venv/bin/activate && python kernels/triton/$(KERNEL).py

clean:
	rm -rf build
