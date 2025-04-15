#include <iostream>
#include <cuda_runtime.h>

#define N 100  // Global grid size
#define NUM_BLOCKS 4  // Number of blocks
#define BLOCK_SIZE 50  // Size of each block (50x50)

struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, width;
    float* localGrid;  // Points to a subregion in the global grid

    __host__ __device__
    BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, int gridWidth = 0,float* gridPtr = nullptr)
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), width(gridWidth), localGrid(gridPtr) {}

    // Utility functions
    __device__ float& at(int i, int j) {
        //int width = yMax - yMin;  // The width of this block
        return localGrid[i * width + j];
    }

    __device__ inline int d_max(int a, int b) {
        return a > b ? a : b;
    }

    __device__ inline int d_min(int a, int b) {
        return a < b ? a : b;
    }

    // Compute method that adds 1.0f to all cells in the local grid
    __device__ void compute() {
        for (int i = 0; i < (xMax - xMin); ++i) {  // Loop over rows
            for (int j = 0; j < (yMax - yMin); ++j) {  // Loop over columns
                at(i, j) = at(i, j) + 1.0;  // Add 1.0f to each element
            }
        }
    }
};

// Kernel: Process blocks, call compute method to add 1.0f
__global__ void processBlocks(BlockOfGrid* blocks, float* deviceLocalGrids, int numBlocks) {
    int idx = threadIdx.x;
    if (idx < numBlocks) {
        blocks[idx].compute();  // Call the compute method for the block
    }
}

int main() {
    // Define 4 blocks (2x2 split of 100x100)
    BlockOfGrid hostBlocks[NUM_BLOCKS]; // Array of type BlockOfGrids in host representing the stack of sub grids
    float* deviceLocalGrids; // Pointer to the 4 local grids (contiguous)
    float* mainGrid = new float[N * N]; // Global grid on host

    //Initialization of mainGrid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0 || j == 0 || i == N || j == N){
                mainGrid[i * N + j] = 22.0f;
            }else{
                mainGrid[i * N + j] = 0.0f;
            }
        }
    }
    //std::cout << mainGrid[0] << std::endl;

    // Allocate memory for local grids (4 blocks, each 50x50)
    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);

    // Set block offsets and initialize BlockOfGrid structs
    int offset = 0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        int xMin = (i / 2) * BLOCK_SIZE;
        int xMax = xMin + BLOCK_SIZE;
        int yMin = (i % 2) * BLOCK_SIZE;
        int yMax = yMin + BLOCK_SIZE;
        int width = yMax - yMin;
        hostBlocks[i] = BlockOfGrid(xMin, xMax, yMin, yMax, width, deviceLocalGrids + offset);
        offset += BLOCK_SIZE * BLOCK_SIZE; // Update offset for next block
    }

    for (int b = 0; b < NUM_BLOCKS; ++b) {
        BlockOfGrid& block = hostBlocks[b];
        //Debugging code starts
        std::cout << "Block " << b << ": "
                << "xMin=" << block.xMin << ", xMax=" << block.xMax
                << ", yMin=" << block.yMin << ", yMax=" << block.yMax
                << ", width=" << block.width << "\n";
        
        /*for (int i = 0; i < (block.xMax - block.xMin); ++i) {
            std::cout << "check 1";
            for (int j = 0; j < (block.yMax - block.yMin); ++j) {
                std::cout << "check 1";
                std::cout << "localGrid[" << i << "][" << j << "] = "
                        << block.localGrid[i * block.width + j] << "\n";
            }
        }*/
    }


    std::cout << "Initialized successfully" << std::endl;

    // Copy blocks to device
    BlockOfGrid* deviceBlocks;
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS);
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    // Run kernel with 4 threads (one for each block)
    processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, deviceLocalGrids, NUM_BLOCKS);
    cudaDeviceSynchronize();

    // Copy the results back to the host from deviceLocalGrids
    cudaMemcpy(mainGrid, deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);

    // Print a few sample results
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        int blockOffset = i * BLOCK_SIZE * BLOCK_SIZE;
        std::cout << "Block " << i << " sample (10,10): "
                  << mainGrid[blockOffset + 10 * BLOCK_SIZE + 10] << "\n";
    }
    std::cout << mainGrid[98] << std::endl;

    // Cleanup
    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    delete[] mainGrid;

    return 0;
}
