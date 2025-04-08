#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

struct Vector {
    float x, y, z;

    __device__ float norm() {
        return sqrt((x * x) + (y * y) + (z * z)) + 1;
    }
};

// CUDA kernel that calls the norm function
__global__ void computeNorm(Vector* v, float* result) {
    *result = v->norm();
}

int main() {
    Vector v1 = {1.0f, 2.0f, 3.0f};
    float *d_result, h_result;
    Vector *d_v1;

    // Allocate memory on the device for the vector and the result
    cudaMalloc((void**)&d_v1, sizeof(Vector));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy the vector to the device
    cudaMemcpy(d_v1, &v1, sizeof(Vector), cudaMemcpyHostToDevice);

    // Launch the kernel to compute the norm
    computeNorm<<<1, 1>>>(d_v1, d_result);

    // Copy the result back to the host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Norm of the vector: " << h_result << std::endl;

    // Free the device memory
    cudaFree(d_v1);
    cudaFree(d_result);

    return 0;
}
