#include <iostream>

__global__ void idcheck() {

    // Correctly formatted printf statement
    printf("Test from blockIdx.x = %d; threadIdx.x = %d; blockIdx.y = %d; threadIdx.y = %d; blockDim.x = %d; blockDim.y = %d; gridDim.x = %d; gridDim.y = %d\n", 
           blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
}


int main() {
    dim3 blocksPerGrid(2, 3);  // 2 blocks in x and y dimensions
    dim3 threadsPerBlock(4, 5); // 4 threads in x and y dimensions

    idcheck<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // Ensure all threads finish before exiting

    return 0;
}

