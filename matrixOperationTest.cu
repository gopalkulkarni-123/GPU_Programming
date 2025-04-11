#include <iostream>
#include <cuda_runtime.h>

#define N 100  // Matrix dimension

// Struct to define a block (submatrix) with bounds
struct MatrixBlock {
    int xMin, xMax, yMin, yMax;

    __device__ void update(float matrix[N][N]) {
        for (int i = xMin; i < xMax; ++i) {
            for (int j = yMin; j < yMax; ++j) {
                matrix[i][j] += 1.0f;
            }
        }
    }
};

// CUDA kernel: each thread handles one block
__global__ void processBlocks(MatrixBlock* blocks, float matrix[N][N]) {
    int idx = threadIdx.x;
    if (idx < 4) {
        blocks[idx].update(matrix);
    }
}

int main() {
    float hostMatrix[N][N];
    float (*deviceMatrix)[N];
    MatrixBlock hostBlocks[4] = {
        {0, 50, 0, 50},     // Top-left
        {0, 50, 50, 100},   // Top-right
        {50, 100, 0, 50},   // Bottom-left
        {50, 100, 50, 100}  // Bottom-right
    };

    // Initialize host matrix with zeros
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            hostMatrix[i][j] = i + j;

    // Allocate device memory
    MatrixBlock* deviceBlocks;
    cudaMalloc(&deviceBlocks, sizeof(MatrixBlock) * 4);
    cudaMalloc(&deviceMatrix, sizeof(float) * N * N);

    // Copy data to device
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(MatrixBlock) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // Launch kernel: 4 threads for 4 blocks
    processBlocks<<<1, 4>>>(deviceBlocks, deviceMatrix);

    // Copy result back
    cudaMemcpy(hostMatrix, deviceMatrix, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    // Print a sample from each block
    std::cout << "Sample from top-left: " << hostMatrix[10][10] << std::endl;
    std::cout << "Sample from top-right: " << hostMatrix[10][60] << std::endl;
    std::cout << "Sample from bottom-left: " << hostMatrix[60][10] << std::endl;
    std::cout << "Sample from bottom-right: " << hostMatrix[60][60] << std::endl;

    // Cleanup
    cudaFree(deviceBlocks);
    cudaFree(deviceMatrix);

    return 0;
}
