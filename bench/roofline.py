from dataclasses import dataclass

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

# --- helpers ----


def cores_per_sm(major, minor):
    # Conservative mapping: Ampere (8.x), Ada (8.9) -> 128 FP32 cores / SM
    # (Good for FP32 “CUDA core” roof. Tensor-core peaks are not derived here.)
    if major >= 8:
        return 128
    # fallback (Turing and earlier)
    tbl = {
        (7, 5): 64,  # Turing TU1xx
        (7, 0): 64,  # Volta GV100-ish FP32 cores usable
        (6, 1): 128,  # Pascal GP10x
        (6, 0): 64,  # Pascal GP100 FP32
        (5, 2): 128,  # Maxwell GM20x
        (5, 0): 128,  # Maxwell GM10x
    }
    return tbl.get((major, minor), 64)


def dtype_bytes(dtype):
    return {
        "fp32": 4,
        "tf32": 4,  # storage 4B; math differs but bandwidth is 4B/elem
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }[dtype]


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
    # torch for SM count + CC; NVML for clocks/bus/mem
    p = torch.cuda.get_device_properties(device_index)
    nvmlInit()
    try:
        h = nvmlDeviceGetHandleByIndex(device_index)
        name_raw = nvmlDeviceGetName(h)
        # Handle both string and bytes return types
        name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
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


def fp32_core_roof_TFLOPs(caps: GPUCaps, use_max_clock=True):
    MHz = caps.sm_clock_max_MHz if use_max_clock else caps.sm_clock_cur_MHz
    cores_sm = cores_per_sm(caps.cc_major, caps.cc_minor)
    # FLOPs = SMs * cores/SM * 2 (FMA) * clock(Hz)
    flops = caps.sm_count * cores_sm * 2 * (MHz * 1e6)
    return flops / 1e12


def dram_bandwidth_GBs(caps: GPUCaps, use_max_clock=True):
    # DDR/GDDR: 2 transfers per cycle
    MHz = caps.mem_clock_max_MHz if use_max_clock else caps.mem_clock_cur_MHz
    bytes_per_cycle = caps.mem_bus_bits / 8
    bw = 2 * (MHz * 1e6) * bytes_per_cycle  # bytes/s
    return bw / 1e9  # GB/s


def gemm_arithmetic_intensity(M, N, K, bytes_per_elem):
    # AI = FLOPs / Bytes moved
    # FLOPs ~ 2*M*N*K
    # Bytes (naive) ~ (M*K + K*N + M*N) * bytes_per_elem  (no cache reuse assumed)
    # For roofline lower bound, this is fine; achieved can be higher with reuse.
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * bytes_per_elem
    return flops / bytes_moved  # FLOPs per byte


def roofline_for_shapes(caps: GPUCaps, shapes, dtype):
    bpe = dtype_bytes(dtype)
    comp_roof = fp32_core_roof_TFLOPs(
        caps, use_max_clock=True
    )  # FP32 core compute roof
    mem_bw = dram_bandwidth_GBs(caps, use_max_clock=True)
    rows = []
    for sh in shapes:
        if isinstance(sh, dict):
            M, N, K = sh["M"], sh["N"], sh["K"]
            B = sh.get("B", 1)
        else:
            B, M, N, K = sh
        ai = gemm_arithmetic_intensity(M, N, K, bpe)
        # Bandwidth roof in TFLOPs = (GB/s * 1e9) * AI / 1e12
        bw_roof_tflops = (mem_bw * 1e9) * ai / 1e12
        rows.append(
            {
                "shape": f"B={B} M={M} N={N} K={K}",
                "dtype": dtype,
                "arith_intensity (FLOP/byte)": ai,
                "bandwidth_roof (TFLOPs)": bw_roof_tflops,
                "fp32_core_roof (TFLOPs)": comp_roof,
                "bottleneck": "memory" if bw_roof_tflops < comp_roof else "compute",
            }
        )
    return rows


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available in torch runtime"
    caps = query_gpu_caps(0)
    print(
        f"GPU: {caps.name}  CC {caps.cc_major}.{caps.cc_minor}  SMs={caps.sm_count}  VRAM={caps.vram_GiB:.1f} GiB"
    )
    print(f"SM clock (cur/max): {caps.sm_clock_cur_MHz} / {caps.sm_clock_max_MHz} MHz")
    print(
        f"Mem clock (cur/max): {caps.mem_clock_cur_MHz} / {caps.mem_clock_max_MHz} MHz"
    )
    print(f"Bus width: {caps.mem_bus_bits} bits")

    fp32_roof = fp32_core_roof_TFLOPs(caps, use_max_clock=True)
    bw = dram_bandwidth_GBs(caps, use_max_clock=True)
    print(f"\nTheoretical FP32 core roof: {fp32_roof:.2f} TFLOPs")
    print(f"Theoretical DRAM bandwidth: {bw:.1f} GB/s")

    # Your batched GEMM shapes to sanity-check (you can edit freely)
    shapes = [
        {"B": 64, "M": 1024, "N": 1024, "K": 1024},
        {"B": 1, "M": 4096, "N": 4096, "K": 4096},
    ]

    for dt in ["fp16", "bf16", "tf32", "fp32"]:
        rows = roofline_for_shapes(caps, shapes, dtype=dt)
        print(f"\n=== Roofline (dtype={dt}) ===")
        for r in rows:
            print(
                f"{r['shape']}: AI={r['arith_intensity (FLOP/byte)']:.2f}, "
                f"BW roof={r['bandwidth_roof (TFLOPs)']:.2f}, "
                f"FP32 roof={r['fp32_core_roof (TFLOPs)']:.2f} → {r['bottleneck']}"
            )
