#include <iostream>
#include <cuda_runtime.h>

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

    __device__ void compute() {
        for (int i = 0; i < (xMax - xMin); ++i) {
            for (int j = 0; j < (yMax - yMin); ++j) {
                at(i, j) = at(i, j) + 2.0f;
            }
        }
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks) {
    int idx = threadIdx.x;
    if (idx < numBlocks) {
        blocks[idx].compute();
    }
}

int main() {
    float* mainGrid = new float[N * N];
    float* hostLocalGrids = new float[NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE];

    // Initialize mainGrid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0 || j == 0 || i == N - 1 || j == N - 1) {
                mainGrid[i * N + j] = 100.0f;
            } else {
                mainGrid[i * N + j] = 0.0f;
            }
        }
    }

    for (int i = 0; i < 100 * 100; ++i){
        std::cout << "mainGrid ["<< i <<"] " << mainGrid[i] << std::endl;
    }
    std::cout << "--------------------------------------------";

    // Define block metadata and copy data from mainGrid to hostLocalGrids
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
                int globalI = xMin + i;
                int globalJ = yMin + j;
                localGridPtr[i * BLOCK_SIZE + j] = mainGrid[globalI * N + globalJ];
            }
        }

        hostBlocks[b] = BlockOfGrid(xMin, xMax, yMin, yMax, BLOCK_SIZE, localGridPtr);
    }

    // Allocate memory on device
    float* deviceLocalGrids;
    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMemcpy(deviceLocalGrids, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice);

    BlockOfGrid* deviceBlocks;
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        hostBlocks[b].localGrid = deviceLocalGrids + b * BLOCK_SIZE * BLOCK_SIZE;
    }
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS);
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    // Launch kernel
    processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, NUM_BLOCKS);
    cudaDeviceSynchronize();

    // Copy back result
    cudaMemcpy(hostLocalGrids, deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);

    // Copy back results to mainGrid for visualization
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        int xMin = hostBlocks[b].xMin;
        int yMin = hostBlocks[b].yMin;
        float* localGridPtr = &hostLocalGrids[b * BLOCK_SIZE * BLOCK_SIZE];

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int globalI = xMin + i;
                int globalJ = yMin + j;
                mainGrid[globalI * N + globalJ] = localGridPtr[i * BLOCK_SIZE + j];
            }
        }
    }

    // Print sample 
    /*for (int i = 0; i < NUM_BLOCKS; ++i) {
        std::cout << "Block " << i << " sample (10,10): "
                  << hostLocalGrids[i * BLOCK_SIZE * BLOCK_SIZE + 10 * BLOCK_SIZE + 10] << "\n";
    }*/
    for (int i = 0; i < 100 * 100; ++i){
        std::cout << "Grid ["<< i <<"] " << hostLocalGrids[i] << std::endl;
    }

    // Cleanup
    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    delete[] hostLocalGrids;
    delete[] mainGrid;

    return 0;
}
