#include <iostream>
#include <cuda_runtime.h>

__global__ void offsetDevice(float* testArray, int offsetValue, int numOfElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + offsetValue;
    testArray[i] = testArray[i] * 2;
}

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

int main() {
    int nMB = 4;
    int n = nMB * 1024 * 1024 / sizeof(float);
    int blockSize = 256;

    cudaEvent_t startEvent, stopEvent;
    float ms;

    //int* hostArray = new int[n];

    //for (int i = 0; i < n; ++i) {
    //    hostArray[i] = i;
    //}

    // Allocate, copy, execute, and free the device memory
    float* deviceArray;
    int size = n * 33 * sizeof(float);
    int numBlocks = (n + blockSize - 1) / blockSize;

    checkCuda(cudaMalloc(&deviceArray, size));

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    offsetDevice<<<numBlocks, blockSize>>>(deviceArray, 0, n);

    for (int i = 0; i < 34; ++i) {
        checkCuda(cudaMemset(deviceArray, 0, n * sizeof(float))); // Corrected memset size

        checkCuda(cudaEventRecord(startEvent, 0));
        offsetDevice<<<numBlocks, blockSize>>>(deviceArray, i, n);
        checkCuda(cudaEventRecord(stopEvent, 0));
        checkCuda(cudaEventSynchronize(stopEvent));

        checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

        std::cout << (2 * n * sizeof(int) / (1024.0f * 1024.0f * 1024.0f)) / (ms / 1000.0f) << std::endl;
    }

    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    checkCuda(cudaFree(deviceArray));
    //delete[] hostArray;

    return 0;
}
