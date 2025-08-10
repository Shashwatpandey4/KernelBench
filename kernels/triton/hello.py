import torch
import triton
import triton.language as tl


@triton.jit
def hello_kernel(Y, N: tl.constexpr):
    i = tl.program_id(0)
    if i < N:
        tl.store(Y + i, i)  # write something predictable


def main(n=8):
    y = torch.empty(n, device="cuda", dtype=torch.int32)
    hello_kernel[(n,)](y, N=n)  # 1 program per element
    torch.cuda.synchronize()
    print("Triton says hello! y =", y.cpu().tolist())


if __name__ == "__main__":
    main()
