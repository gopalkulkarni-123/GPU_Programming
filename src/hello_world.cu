#include <iostream>

__global__ void cuda_hello(){
    //printf("Hello World from GPU! \n");
    printf("Grid Dim :%d \n", gridDim.x);
    printf("Block Dim :%d \n", blockDim.x);
}

int main(){
    cuda_hello<<<9, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}