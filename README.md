# KernelBench

GPU kernel benchmarking with roofline model analysis for CUDA, Triton, and Mojo implementations.

## Overview

Benchmarking suite for batched GEMM kernels across multiple GPU programming languages. Includes roofline model visualization to analyze performance against theoretical hardware limits.

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 8.0+
- Tested on: NVIDIA GeForce RTX 4060 Laptop GPU (CC 8.9, 24 SMs, 8GB VRAM)

### Software
- CUDA Toolkit 12.8+
- NVIDIA Driver 560.35.05+
- Python 3.13+
- uv (Python package manager)

## Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

### Roofline Analysis

```bash
# Basic roofline plot
python bench/roofline_bgemm.py --dtype fp16

# Include cuBLAS baseline measurements
python bench/roofline_bgemm.py --dtype fp16 --calibrate

# Different data types
python bench/roofline_bgemm.py --dtype fp32 --calibrate
python bench/roofline_bgemm.py --dtype tf32 --calibrate
```

### Benchmark Results (RTX 4060 Laptop)

| Data Type | cuBLAS Performance | Theoretical Peak | Efficiency |
|-----------|-------------------|------------------|------------|
| FP16      | ~28 TFLOPs        | 76.3 TFLOPs     | 37%        |
| FP32      | ~8 TFLOPs         | 19.1 TFLOPs     | 42%        |

All tested workloads are compute-bound with high arithmetic intensity (170-1365 FLOPs/byte).

## Project Structure

```
KernelBench/
├── bench/
│   ├── roofline_bgemm.py    # Main roofline analysis tool
│   └── roofline.py          # GPU capability detection
├── kernels/
│   ├── cuda/                # CUDA kernel implementations
│   ├── triton/              # Triton kernel implementations
│   └── mojo/                # Mojo kernel implementations
└── build/                   # Generated plots and artifacts
```

## Adding Kernel Implementations

1. Implement kernel in appropriate language directory
2. Add timing function to `roofline_bgemm.py`:
   ```python
   def time_your_kernel_bgemm(B, M, N, K, dtype: str) -> float:
       # Implementation
       return tflops_measured
   ```
3. Run roofline analysis to compare performance

## Dependencies

Managed via `uv` - see `pyproject.toml` for complete list:
- torch (PyTorch for cuBLAS baseline)
- matplotlib (plotting)
- numpy (numerical operations)
- pynvml (GPU monitoring) 