#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define N 100  // Global grid size
#define NUM_BLOCKS 4  // Number of blocks
#define BLOCK_SIZE 50  // Size of each block (50x50)

struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, width;
    float* localGrid;  // Points to a subregion in the global grid

    __host__ __device__
    BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, int gridWidth = 0, float* gridPtr = nullptr)
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), width(gridWidth), localGrid(gridPtr) {}

    // Utility functions
    __device__ float& at(int i, int j) {
        return localGrid[i * width + j];
    }

    __device__ inline int d_max(int a, int b) {
        return a > b ? a : b;
    }   

    __device__ inline int d_min(int a, int b) {
        return a < b ? a : b;
    }

    __device__ void compute(float* dGrid) {

    /*for (int i = 0; i < 100 * 100; ++i){
        printf("dGrid[%d] = %f \n", i, dGrid[i]);
    }*/

        for (int i = 0; i < (xMax - xMin); ++i) {
            for (int j = 0; j < (yMax - yMin); ++j) {
                localGrid[i * width + j] = dGrid[(xMin + i) * 100 + (yMin + j)] + 1.0f;
            }
        }
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks, float* Grid) {

    int idx = threadIdx.x;
    
    if (idx < numBlocks) {
        blocks[idx].compute(Grid);
    }

    // Now populate the results back into mainGrid
        BlockOfGrid& block = blocks[idx];  // Reference to the current block
        for (int i = block.xMin; i < block.xMax; ++i) {
            for (int j = block.yMin; j < block.yMax; ++j) {
                Grid[i * 100 + j] = block.localGrid[(i - block.xMin) * block.width + (j - block.yMin)];
            }
        }
}

/*__global__ void reduceMaxKernel(float* d_input, float* d_output, void* d_temp_storage, size_t temp_storage_bytes, int numItems) {
    // Let only one thread do it (for simplicity)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, numItems);
    }
}*/


int main() {
    float* mainGrid = new float[N * N];
    float* hostLocalGrids = new float[NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE];

    // Initialize mainGrid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0 || j == 0 || i == N - 1 || j == N - 1) {
                mainGrid[(i * N) + j] = 100.0f;
            } else {
                mainGrid[(i * N) + j] = 0.0f;
            }
        }
    }
    mainGrid[(50 * 100 + 50)] = 205.0f;

    BlockOfGrid hostBlocks[NUM_BLOCKS];
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        int xMin = (b / 2) * BLOCK_SIZE;
        int xMax = xMin + BLOCK_SIZE;
        int yMin = (b % 2) * BLOCK_SIZE;
        int yMax = yMin + BLOCK_SIZE;
        int width = yMax - yMin;
        float* localGridPtr = &hostLocalGrids[b * BLOCK_SIZE * BLOCK_SIZE];

        // Copy corresponding block from mainGrid
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {

                localGridPtr[i * BLOCK_SIZE + j] = mainGrid[(b * BLOCK_SIZE * BLOCK_SIZE) + (i * BLOCK_SIZE + j)];
            }
        }

        hostBlocks[b] = BlockOfGrid(xMin, xMax, yMin, yMax, BLOCK_SIZE, localGridPtr);
    }

    // Allocate memory on device
    float* deviceLocalGrids;
    float* deviceMainGrid;

    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMalloc(&deviceMainGrid, sizeof(float) * N * N);
    
    cudaMemcpy(deviceLocalGrids, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice); //Check the syntax
    cudaMemcpy(deviceMainGrid, mainGrid, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    BlockOfGrid* deviceBlocks;
    for (int b = 0; b < NUM_BLOCKS; ++b) {

        hostBlocks[b].localGrid = deviceLocalGrids + b * BLOCK_SIZE * BLOCK_SIZE;
    }
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS);
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    //Memory allocation for finding the maximum
    /*float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMemcpy(d_input, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, sizeof(float));

    // 2. Temporary storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temp storage size
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output,
                        NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);*/

    for (int i = 0; i < 5; ++i){
        // Launch kernel
        processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, NUM_BLOCKS, deviceMainGrid);
        cudaDeviceSynchronize();

        /*reduceMaxKernel<<<1, 32>>>(d_input, d_output, d_temp_storage, temp_storage_bytes, NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
        cudaDeviceSynchronize();*/

        // Copy back result
        cudaMemcpy(hostLocalGrids, deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(mainGrid, deviceMainGrid, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        
        /*cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output,
                            NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);


        float maxValue;
        cudaMemcpy(&maxValue, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "\n Max grid value across all blocks: " << maxValue << "\n";*/

        // Displays the main grid
        for (int j = 0; j < 100 * 100; ++j){
            std::cout << "processedGrid ["<< j <<"] " << mainGrid[j] << " at iter " << i << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "processedGrid [5050] = " << mainGrid[5050] << " at iter " << i << std::endl;

    }

    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    delete[] hostLocalGrids;
    delete[] mainGrid;

    // Cleanup CUB allocations
    /*cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);*/

    return 0;
}