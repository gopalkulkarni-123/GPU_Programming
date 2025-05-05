#include <iostream>
#include <math.h>

__global__ void add_vectors(float *x, float *y, int n){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride){
        y[i] = x[i] + y[i];
    }
}

int main(){
    int N = 1<<20;
    float *vec_1, *vec_2;
    int blockSize = 256;
    int numOfBlocks;

    cudaMallocManaged (&vec_1, N*sizeof(float));
    cudaMallocManaged (&vec_2, N*sizeof(float));

    for (int i = 0; i < N; ++i){
        vec_1[i] = 1.0f;
        vec_2[i] = 7.0f;
    }
    
    numOfBlocks = (N + blockSize - 1)/blockSize;

    add_vectors<<<numOfBlocks, blockSize>>>(vec_1, vec_2, N);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; ++i){
        maxError = fmax(maxError, fabs(vec_2[i] - 8.0f));
    }
    
    std::cout << "Max error : " << maxError << std::endl;

    cudaFree(vec_1);
    cudaFree(vec_2);

    return 0;
    }