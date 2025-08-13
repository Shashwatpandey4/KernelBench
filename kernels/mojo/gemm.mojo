from time import perf_counter
from random import random_float64


def sgemm_naive(
    A: List[Float16],
    B: List[Float16],
    mut C: List[Float16],
    M: Int,
    N: Int,
    K: Int,
    alpha: Float64 = 1.0,
    beta: Float64 = 0.0,
):
    for i in range(M):
        baseA = i * K
        baseC = i * N
        for j in range(N):
            var acc: Float64 = 0.0
            for k in range(K):
                acc += Float64(A[baseA + k]) * Float64(B[k * N + j])
            C[baseC + j] = Float16(alpha * acc + beta * Float64(C[baseC + j]))


def main():
    M = 1024
    N = 1024
    K = 1024
    alpha = 1.0
    beta = 0.0

    var A = List[Float16]()
    var B = List[Float16]()
    var C = List[Float16]()

    for _ in range(M * K):
        A.append(Float16(random_float64()))
    for _ in range(K * N):
        B.append(Float16(random_float64()))
    for _ in range(M * N):
        C.append(Float16(0.0))

    # warm-up (optional)
    sgemm_naive(A, B, C, M, N, K, alpha, beta)

    # time it
    start = perf_counter()
    sgemm_naive(A, B, C, M, N, K, alpha, beta)
    t = perf_counter() - start

    # FLOPs and throughput
    flops = 2.0 * M * N * K  # multiply + add per MAC
    gflops = flops / t / 1e9

    print("M,N,K=", M, ",", N, ",", K)
    print("time =", t, "s")
    print("FLOPs =", flops)
    print("Throughput =", gflops, "GFLOP/s")
