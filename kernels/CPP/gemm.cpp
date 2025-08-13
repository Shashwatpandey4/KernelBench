#include <iostream>
#include <vector>
#include <ctime>

const int K = 1024;
const int M = 1024;
const int N = 1024;

// Naive matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
void naive_gemm(const std::vector<std::vector<float>> &A,
                const std::vector<std::vector<float>> &B,
                std::vector<std::vector<float>> &C)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0.0f;
            for (int k = 0; k < K; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Initialize matrix with simple values
void init_matrix(std::vector<std::vector<float>> &matrix, int rows, int cols, float base_value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = base_value + (i * cols + j) * 0.001f;
        }
    }
}

int main()
{
    std::cout << "Naive GEMM - " << M << "x" << K << " * " << K << "x" << N << std::endl;

    // Allocate matrices
    std::vector<std::vector<float>> A(M, std::vector<float>(K));
    std::vector<std::vector<float>> B(K, std::vector<float>(N));
    std::vector<std::vector<float>> C(M, std::vector<float>(N));

    std::cout << "Initializing matrices..." << std::endl;
    init_matrix(A, M, K, 1.0f);
    init_matrix(B, K, N, 2.0f);

    std::cout << "Starting matrix multiplication..." << std::endl;

    // Time the matrix multiplication
    clock_t start = clock();
    naive_gemm(A, B, C);
    clock_t end = clock();

    // Calculate timing and performance
    double time_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    long long flops = 2LL * M * N * K;                    // 2 operations (multiply + add) per inner loop iteration
    double gflops = (double)flops / (time_seconds * 1e9); // Convert to GFLOPS

    std::cout << "Matrix multiplication completed!" << std::endl;
    std::cout << "Time: " << time_seconds << " seconds" << std::endl;
    std::cout << "FLOPS: " << flops << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;

    // Simple verification - check first element
    std::cout << "First element C[0][0] = " << C[0][0] << std::endl;
    std::cout << "Last element C[1023][1023] = " << C[1023][1023] << std::endl;

    return 0;
}
