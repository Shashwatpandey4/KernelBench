SHELL := /bin/bash

NVCC := /usr/local/cuda-12.8/bin/nvcc
GPU_ARCH ?= sm_86
HOST_CXX ?= /usr/bin/g++-12
NVCCFLAGS := -O3 --use_fast_math -std=c++14 -arch=$(GPU_ARCH) -ccbin=$(HOST_CXX) \
             -cudart shared -Xlinker -rpath,/usr/local/cuda-12.8/lib64

MOJOENV := .mojoenv
MOJOFLAGS := -O3

KERNEL ?= hello

.PHONY: run_triton run_cuda run_mojo clean

build/cuda/%: kernels/cuda/%.cu
	@mkdir -p build/cuda
	$(NVCC) $(NVCCFLAGS) $< -o $@

run_cuda: build/cuda/$(KERNEL)
	@./build/cuda/$(KERNEL)

run_mojo:
	@cd $(MOJOENV) && pixi run mojo run ../kernels/mojo/$(KERNEL).mojo

run_triton:
	@. .venv/bin/activate && python kernels/triton/$(KERNEL).py

clean:
	rm -rf build
