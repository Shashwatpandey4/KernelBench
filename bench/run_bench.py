# bench/run_bench.py
import time

import click
import pandas as pd

from .utils import run_cuda_kernel, run_mojo_bin, run_triton_kernel

CASES = [
    ("bias_gelu", {"shape": (4096, 4096), "dtype": "fp16"}),
]

IMPLS = {
    "triton": run_triton_kernel,
    "cuda": run_cuda_kernel,
    "mojo": run_mojo_bin,
}


def bench_one(kind, name, params, repeats=50, warmup=10):
    fn = IMPLS[kind]
    for _ in range(warmup):
        fn(name, params, sync=True)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(name, params, sync=True)
        times.append((time.perf_counter() - t0) * 1000)
    s = pd.Series(times)
    return dict(
        kernel=name,
        impl=kind,
        median_ms=float(s.median()),
        p10_ms=float(s.quantile(0.10)),
        p90_ms=float(s.quantile(0.90)),
        n=repeats,
    )


@click.command()
@click.option("--impls", default="triton,cuda,mojo", help="comma list")
@click.option("--only", default="", help="optional kernel name filter")
def main(impls, only):
    impls = [x.strip() for x in impls.split(",")]
    rows = []
    for name, params in CASES:
        if only and name != only:
            continue
        for kind in impls:
            rows.append(bench_one(kind, name, params))
    df = pd.DataFrame(rows)
    out = "build/results.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
