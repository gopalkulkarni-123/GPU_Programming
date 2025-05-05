#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

struct Vector {
    float x, y, z;

    __device__ float norm() {
        return sqrt((x * x) + (y * y) + (z * z));
    }
};

// CUDA kernel to compute norms for a stack of vectors
__global__ void computeNorms(Vector* v, float* result, int numVectors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    if (idx < numVectors) {
        result[idx] = v[idx].norm();  // Compute the norm for the corresponding vector
    }
}

int main() {
    int numVectors = 1000000000;  // 1 billion vectors
    size_t vectorSize = sizeof(Vector);
    size_t resultSize = sizeof(float) * numVectors;
    float *d_result, *h_result;
    Vector *d_v1;
    
    // Allocate memory for the vectors and result on the device
    cudaMalloc((void**)&d_v1, vectorSize * numVectors);
    cudaMalloc((void**)&d_result, resultSize);

    // Create a stack of vectors on the host (initialize them)
    std::vector<Vector> h_v1(numVectors, {1.0f, 2.0f, 3.0f});  // Example vectors

    // Copy the vectors to the device
    cudaMemcpy(d_v1, h_v1.data(), vectorSize * numVectors, cudaMemcpyHostToDevice);

    // Allocate host memory for the results
    h_result = new float[numVectors];

    // Set up the execution configuration
    int blockSize = 256;  // Threads per block
    int numBlocks = (numVectors + blockSize - 1) / blockSize;  // Calculate number of blocks

    // Launch the kernel
    computeNorms<<<numBlocks, blockSize>>>(d_v1, d_result, numVectors);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Print the first few norms (just to check)
    std::cout << "Norm of the first vector: " << h_result[0] << std::endl;
    std::cout << "Norm of the second vector: " << h_result[1] << std::endl;
    std::cout << "Norm of the second vector: " << h_result[1000000000 - 1] << std::endl;

    // Clean up memory
    cudaFree(d_v1);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}