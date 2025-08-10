// kernels/cuda/hello.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(float *a, float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    printf("Hello from CUDA!\n");

    // Simple CUDA runtime check with detailed error reporting
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        printf("CUDA error in cudaSetDevice: %s (code %d)\n", cudaGetErrorString(err), err);
        printf("This might be due to:\n");
        printf("1. No CUDA-capable GPU found\n");
        printf("2. CUDA driver/runtime version mismatch\n");
        printf("3. GPU in use by another process\n");
        printf("4. Insufficient permissions\n");
        return 1;
    }

    printf("CUDA device 0 set successfully!\n");

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
    {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // If we get here, try a simple vector addition
    const int N = 8;
    const size_t bytes = N * sizeof(float);

    // Host arrays
    float h_a[N], h_b[N], h_c[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
        h_c[i] = 0.0f; // Initialize output
    }

    printf("Attempting GPU computation...\n");

    // Device arrays
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;

    // Variables declared here to avoid goto scope issues
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Try to allocate device memory
    err = cudaMalloc(&d_a, bytes);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&d_b, bytes);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_b: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return 1;
    }

    err = cudaMalloc(&d_c, bytes);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    printf("GPU memory allocated successfully!\n");

    // Copy data to device
    err = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed for d_a: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed for d_b: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    printf("Data copied to GPU successfully!\n");

    // Launch kernel
    printf("Launching kernel with grid=%d, block=%d\n", gridSize, blockSize);

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    printf("Kernel executed successfully!\n");

    // Copy result back to host
    err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy back to host failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Print results
    printf("Vector Addition Results:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }

    printf("CUDA vector addition completed successfully!\n");

cleanup:
    if (d_a)
        cudaFree(d_a);
    if (d_b)
        cudaFree(d_b);
    if (d_c)
        cudaFree(d_c);

    return (err == cudaSuccess) ? 0 : 1;
}
