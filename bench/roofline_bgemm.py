#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from pynvml import (
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMaxClockInfo,
    nvmlDeviceGetMemoryBusWidth,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)

# ---------------- GPU caps + roofs ----------------


def cores_per_sm(major: int, minor: int) -> int:
    # FP32 "CUDA cores" per SM (for FP32 roof). Ada (8.9) uses 128.
    if major >= 8:
        return 128
    table = {(7, 5): 64, (7, 0): 64, (6, 1): 128, (6, 0): 64, (5, 2): 128, (5, 0): 128}
    return table.get((major, minor), 64)


@dataclass
class GPUCaps:
    name: str
    sm_count: int
    cc_major: int
    cc_minor: int
    sm_clock_max_MHz: int
    sm_clock_cur_MHz: int
    mem_clock_max_MHz: int
    mem_clock_cur_MHz: int
    mem_bus_bits: int
    vram_GiB: float


def query_gpu_caps(device_index=0) -> GPUCaps:
    p = torch.cuda.get_device_properties(device_index)
    nvmlInit()
    try:
        h = nvmlDeviceGetHandleByIndex(device_index)
        name = nvmlDeviceGetName(h)
        sm_clock_max = nvmlDeviceGetMaxClockInfo(h, NVML_CLOCK_SM)
        sm_clock_cur = nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM)
        mem_clock_max = nvmlDeviceGetMaxClockInfo(h, NVML_CLOCK_MEM)
        mem_clock_cur = nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM)
        bus_bits = nvmlDeviceGetMemoryBusWidth(h)
        mem = nvmlDeviceGetMemoryInfo(h)
        vram_GiB = mem.total / (1024**3)
    finally:
        nvmlShutdown()
    return GPUCaps(
        name=name,
        sm_count=p.multi_processor_count,
        cc_major=p.major,
        cc_minor=p.minor,
        sm_clock_max_MHz=sm_clock_max,
        sm_clock_cur_MHz=sm_clock_cur,
        mem_clock_max_MHz=mem_clock_max,
        mem_clock_cur_MHz=mem_clock_cur,
        mem_bus_bits=bus_bits,
        vram_GiB=vram_GiB,
    )


def fp32_core_roof_TFLOPs(caps: GPUCaps, use_max_clock=True) -> float:
    MHz = caps.sm_clock_max_MHz if use_max_clock else caps.sm_clock_cur_MHz
    flops = caps.sm_count * cores_per_sm(caps.cc_major, caps.cc_minor) * 2 * (MHz * 1e6)
    return flops / 1e12


# dtype -> bytes per element (for memory traffic)
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "int8": 1}

# dtype -> multiplier vs FP32 compute roof (tensor cores)
# Conservative defaults for Ada (SM 89).
DTYPE_TC_MULT = {"fp32": 1.0, "tf32": 2.0, "fp16": 4.0, "bf16": 4.0}


def dtype_compute_roof(fp32_roof_tflops: float, dtype: str) -> float:
    return fp32_roof_tflops * DTYPE_TC_MULT.get(dtype, 1.0)


def dram_bandwidth_GBs(caps: GPUCaps, use_max_clock=True) -> float:
    MHz = caps.mem_clock_max_MHz if use_max_clock else caps.mem_clock_cur_MHz
    # 2 transfers/cycle (GDDR), bus_bits/8 bytes per transfer
    bw_bytes = 2 * (MHz * 1e6) * (caps.mem_bus_bits / 8.0)
    return bw_bytes / 1e9


# --------------- GEMM math ---------------


def gemm_ai_flops_per_byte(M, N, K, dtype: str) -> float:
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * DTYPE_BYTES[dtype]
    return flops / bytes_moved


def tflops_from_time(B, M, N, K, seconds: float) -> float:
    flops = 2.0 * B * M * N * K
    return (flops / seconds) / 1e12


# --------------- cuBLAS timing (via PyTorch) ---------------

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def time_cublas_bgemm(B, M, N, K, dtype: str, repeats=50, warmup=20) -> float:
    assert torch.cuda.is_available()
    device = "cuda"

    if dtype == "tf32":
        torch.set_float32_matmul_precision("high")
        dt = torch.float32
    else:
        dt = TORCH_DTYPES[dtype]

    # Pre-allocate tensors to avoid allocation overhead in timing
    A = torch.randn(B, M, K, device=device, dtype=dt)
    Bm = torch.randn(B, K, N, device=device, dtype=dt)

    # warmup - more extensive for stable measurements
    for _ in range(warmup):
        C = A @ Bm
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()  # Ensure clean start
        t0 = time.perf_counter()
        C = A @ Bm
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # Use more robust statistics - remove outliers and take mean of middle 80%
    times = np.array(times)
    p10, p90 = np.percentile(times, [10, 90])
    filtered_times = times[(times >= p10) & (times <= p90)]
    t = np.mean(filtered_times) if len(filtered_times) > 0 else np.median(times)

    return tflops_from_time(B, M, N, K, t)


# --------------- Placeholder kernel functions ---------------


def time_mojo_bgemm(B, M, N, K, dtype: str) -> float:
    """Placeholder for Mojo kernel - returns 0.0 until implemented"""
    # TODO: Implement Mojo batched GEMM kernel
    return 0.0


def time_cuda_bgemm(B, M, N, K, dtype: str) -> float:
    """Placeholder for custom CUDA kernel - returns 0.0 until implemented"""
    # TODO: Implement custom CUDA batched GEMM kernel
    return 0.0


def time_triton_bgemm(B, M, N, K, dtype: str) -> float:
    """Placeholder for Triton kernel - returns 0.0 until implemented"""
    # TODO: Implement Triton batched GEMM kernel
    return 0.0


# --------------- Plotting ---------------


def plot_roofline(
    shapes: List[Dict],
    dtype: str,
    calibrate: bool,
    outfile: str,
    show_placeholders: bool = True,
) -> None:
    caps = query_gpu_caps(0)
    fp32_roof = fp32_core_roof_TFLOPs(caps, use_max_clock=True)
    comp_roof = dtype_compute_roof(fp32_roof, dtype)
    bw_gbs = dram_bandwidth_GBs(caps, use_max_clock=True)

    ai_range = np.logspace(-1, 4, 1000)
    mem_roof = (bw_gbs * ai_range) / 1000.0
    overall = np.minimum(mem_roof, np.full_like(ai_range, comp_roof))

    # Plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(ai_range, overall, "k-", lw=2, label="Roofline")
    ax.plot(ai_range, mem_roof, "b--", alpha=0.5, label="Memory roof")
    ax.axhline(
        comp_roof, color="r", ls="--", alpha=0.6, label=f"{dtype.upper()} compute roof"
    )

    # Theoretical points
    labels_done = set()
    per_shape_ai = []
    theoretical_points = []
    for s in shapes:
        B, M, N, K = s.get("B", 1), s["M"], s["N"], s["K"]
        ai = gemm_ai_flops_per_byte(M, N, K, dtype)
        per_shape_ai.append(ai)
        perf_bw = (bw_gbs * ai) / 1000.0
        theo = min(perf_bw, comp_roof)
        theoretical_points.append((ai, theo))
        lbl = f"Theoretical: B{B}_M{M}_N{N}_K{K}"
        ax.plot(
            ai, theo, "ko", ms=8, label=lbl if "Theoretical" not in labels_done else ""
        )
        labels_done.add("Theoretical")

    # Measured cuBLAS (optional)
    measured_cublas = []
    cublas_points = []
    if calibrate:
        for s in shapes:
            B, M, N, K = s.get("B", 1), s["M"], s["N"], s["K"]
            tflops = time_cublas_bgemm(B, M, N, K, dtype)
            measured_cublas.append(tflops)
            ai = gemm_ai_flops_per_byte(M, N, K, dtype)
            cublas_points.append((ai, tflops))

        # Plot cuBLAS points and line
        if cublas_points:
            ais, perfs = zip(*cublas_points)
            ax.plot(
                ais,
                perfs,
                "s-",
                ms=7,
                color="tab:red",
                linewidth=2,
                label="cuBLAS (measured)",
            )
            labels_done.add("cuBLAS")
    else:
        measured_cublas = [None] * len(shapes)

    # Initialize kernel measurement arrays
    measured_mojo = [0.0] * len(shapes)
    measured_cuda = [0.0] * len(shapes)
    measured_triton = [0.0] * len(shapes)

    # Collect points for each kernel type
    mojo_points = []
    cuda_points = []
    triton_points = []

    # Kernel placeholders with estimated performance
    if show_placeholders:
        for i, s in enumerate(shapes):
            B, M, N, K = s.get("B", 1), s["M"], s["N"], s["K"]
            ai = gemm_ai_flops_per_byte(M, N, K, dtype)

            # Get actual measurements (or 0.0 if not implemented)
            mojo_tflops = time_mojo_bgemm(B, M, N, K, dtype)
            cuda_tflops = time_cuda_bgemm(B, M, N, K, dtype)
            triton_tflops = time_triton_bgemm(B, M, N, K, dtype)

            measured_mojo[i] = mojo_tflops
            measured_cuda[i] = cuda_tflops
            measured_triton[i] = triton_tflops

            # Handle Mojo - either measured or placeholder
            if mojo_tflops == 0.0:
                # Estimate: 60% of cuBLAS or 40% of theoretical, whichever is lower
                cublas_perf = measured_cublas[i] if measured_cublas[i] else 0
                theo_perf = min((bw_gbs * ai) / 1000.0, comp_roof)
                estimated = min(
                    cublas_perf * 0.6 if cublas_perf > 0 else theo_perf * 0.4,
                    theo_perf * 0.4,
                )
                mojo_points.append((ai, estimated))
            else:
                mojo_points.append((ai, mojo_tflops))

            # Handle CUDA - either measured or placeholder
            if cuda_tflops == 0.0:
                # Estimate: 80% of cuBLAS or 60% of theoretical, whichever is lower
                cublas_perf = measured_cublas[i] if measured_cublas[i] else 0
                theo_perf = min((bw_gbs * ai) / 1000.0, comp_roof)
                estimated = min(
                    cublas_perf * 0.8 if cublas_perf > 0 else theo_perf * 0.6,
                    theo_perf * 0.6,
                )
                cuda_points.append((ai, estimated))
            else:
                cuda_points.append((ai, cuda_tflops))

            # Handle Triton - either measured or placeholder
            if triton_tflops == 0.0:
                # Estimate: 70% of cuBLAS or 50% of theoretical, whichever is lower
                cublas_perf = measured_cublas[i] if measured_cublas[i] else 0
                theo_perf = min((bw_gbs * ai) / 1000.0, comp_roof)
                estimated = min(
                    cublas_perf * 0.7 if cublas_perf > 0 else theo_perf * 0.5,
                    theo_perf * 0.5,
                )
                triton_points.append((ai, estimated))
            else:
                triton_points.append((ai, triton_tflops))

        # Plot kernel lines and points (one legend entry each)
        if mojo_points:
            ais, perfs = zip(*mojo_points)
            has_measured = any(measured_mojo[i] > 0 for i in range(len(shapes)))
            label = "Mojo (measured)" if has_measured else "Mojo (placeholder)"
            alpha = 1.0 if has_measured else 0.6
            ax.plot(
                ais,
                perfs,
                "D-",
                ms=6,
                color="green",
                alpha=alpha,
                linewidth=2,
                label=label,
            )

        if cuda_points:
            ais, perfs = zip(*cuda_points)
            has_measured = any(measured_cuda[i] > 0 for i in range(len(shapes)))
            label = "CUDA (measured)" if has_measured else "CUDA (placeholder)"
            alpha = 1.0 if has_measured else 0.6
            ax.plot(
                ais,
                perfs,
                "^-",
                ms=6,
                color="blue",
                alpha=alpha,
                linewidth=2,
                label=label,
            )

        if triton_points:
            ais, perfs = zip(*triton_points)
            has_measured = any(measured_triton[i] > 0 for i in range(len(shapes)))
            label = "Triton (measured)" if has_measured else "Triton (placeholder)"
            alpha = 1.0 if has_measured else 0.6
            ax.plot(
                ais,
                perfs,
                "o-",
                ms=6,
                color="orange",
                alpha=alpha,
                linewidth=2,
                label=label,
            )

    # Cosmetics
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (TFLOPs)")
    ax.set_title(
        f"Roofline Model - {caps.name}\nBatchedGEMM Performance Analysis ({dtype.upper()})"
    )
    ax.legend(loc="lower right")

    info = (
        f"GPU: {caps.name}\n"
        f"CC: {caps.cc_major}.{caps.cc_minor}\n"
        f"SMs: {caps.sm_count}\n"
        f"VRAM: {caps.vram_GiB:.1f} GiB\n"
        f"FP32 Peak: {fp32_roof:.1f} TFLOPs\n"
        f"Memory BW: {bw_gbs:.0f} GB/s"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved {outfile}")

    # Enhanced CSV output
    print(
        "\nshape, AI(FLOP/byte), cuBLAS_TFLOPs, CUDA_TFLOPs, Triton_TFLOPs, Mojo_TFLOPs, compute_roof_TFLOPs, bw_roof_TFLOPs"
    )
    for i, (s, ai) in enumerate(zip(shapes, per_shape_ai)):
        perf_bw = (bw_gbs * ai) / 1000.0
        cublas_val = measured_cublas[i] if measured_cublas[i] else None
        cuda_val = (
            measured_cuda[i] if show_placeholders and measured_cuda[i] > 0 else None
        )
        triton_val = (
            measured_triton[i] if show_placeholders and measured_triton[i] > 0 else None
        )
        mojo_val = (
            measured_mojo[i] if show_placeholders and measured_mojo[i] > 0 else None
        )

        print(
            f"B{s.get('B',1)}_M{s['M']}_N{s['N']}_K{s['K']}, "
            f"{ai:.2f}, "
            f"{'-' if cublas_val is None else f'{cublas_val:.2f}'}, "
            f"{'-' if cuda_val is None else f'{cuda_val:.2f}'}, "
            f"{'-' if triton_val is None else f'{triton_val:.2f}'}, "
            f"{'-' if mojo_val is None else f'{mojo_val:.2f}'}, "
            f"{comp_roof:.2f}, "
            f"{perf_bw:.2f}"
        )


# --------------- CLI ---------------


def parse_args():
    ap = argparse.ArgumentParser(description="Roofline for Batched GEMM")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "tf32", "fp32"])
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Run cuBLAS (via PyTorch) and overlay measured points; also refines your intuition for the compute roof.",
    )
    ap.add_argument(
        "--out", default=None, help="Output PNG file (default based on dtype)"
    )
    ap.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Hide placeholder points for unimplemented kernels (Mojo, CUDA, Triton)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available for PyTorch"

    # your default shapes; edit as you like
    SHAPES = [
        {"B": 32, "M": 1024, "N": 1024, "K": 1024},
        {"B": 64, "M": 2048, "N": 2048, "K": 2048},
        {"B": 16, "M": 4096, "N": 4096, "K": 4096},
        {"B": 8, "M": 8192, "N": 8192, "K": 1024},
    ]

    args = parse_args()
    out = args.out or f"build/roofline_batchedGEMM_{args.dtype}.png"
    show_placeholders = not args.no_placeholders
    plot_roofline(
        SHAPES,
        dtype=args.dtype,
        calibrate=args.calibrate,
        outfile=out,
        show_placeholders=show_placeholders,
    )
