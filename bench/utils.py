import json
import os
import subprocess


def _sync_cuda():
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def run_triton_kernel(name, params, sync=True):
    if name == "bias_gelu":
        from kernels.triton.bias_gelu import run as triton_run

        triton_run(**params)
    else:
        raise NotImplementedError(name)
    if sync:
        _sync_cuda()


def run_cuda_kernel(name, params, sync=True):
    # placeholder until you wire CUDA launches (via a Python binding or CLI)
    # For now just no-op to keep the bench pipeline happy.
    if sync:
        pass


def run_mojo_bin(name, params, sync=True):
    # placeholder: call built mojo binary with JSON args (match your future CLI)
    binpath = os.path.join("build", "mojo", name)
    if not os.path.exists(binpath):
        return
    subprocess.run([binpath, json.dumps(params)], check=True)
    if sync:
        pass
