#include <stdio.h>

__device__ void addTwo(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = arr[i] + 2.0f;  // Access and modify elements via pointer
    }
}

__global__ void kernel(float* arr, int size) {
    addTwo(arr, size);  // Pass the array pointer to the function
}

int main() {
    const int size = 5;
    float h_arr[size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Allocate device memory
    float* d_arr;
    cudaMalloc(&d_arr, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    kernel<<<1,1>>>(d_arr, size);

    // Copy data back to host
    cudaMemcpy(h_arr, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the updated array
    for (int i = 0; i < size; ++i) {
        printf("%f ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
