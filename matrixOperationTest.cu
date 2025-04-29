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

    __device__ void compute(float* mainGrid) {
        for (int i = 0; i < (xMax - xMin); ++i) {
            for (int j = 0; j < (yMax - yMin); ++j) {
                //localGrid[i * width + j] = mainGrid[(xMin + i) * width + (yMin + j)] + 2.0f;
                printf("Check \n"); //Works!!
                //printf("localGrid index = %d || mainGrid index = %d \n", (i * width + j), ((xMin + i) * width + (yMin + j))); Works!!
                //printf("localGrid [0] = %f \n", localGrid[0]); Works!!   
            }
        }
    }

    __device__ void debug_array(float* mainGrid) {
        for (int i = 0; i < 10; ++i){
            printf("%f \n", mainGrid[i]);
        }
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks, float* mainGrid) {
    int idx = threadIdx.x;
    
    // Ensure that the block index is within range
    if (idx < numBlocks) {
        // Compute the subgrid (localGrid) for this block
        //blocks[idx].compute(mainGrid);
        blocks[idx].debug_array(mainGrid);
        //printf("check");
        
        // Now populate the results back into mainGrid
        /*BlockOfGrid& block = blocks[idx];  // Reference to the current block
        for (int i = block.xMin; i < block.xMax; ++i) {
            for (int j = block.yMin; j < block.yMax; ++j) {
                //mainGrid[i * block.width + j] = block.localGrid[(i - block.xMin) * block.width + (j - block.yMin)];
                printf("check");
            }
        }*/
    }
}


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

    // Displays the main grid
    /*for (int i = 0; i < 100 * 100; ++i){
        std::cout << "mainGrid ["<< i <<"] " << mainGrid[i] << std::endl;
    }
    std::cout << "--------------------------------------------" << std::endl;*/

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

                //int globalI = xMin + i;
                //int globalJ = yMin + j;

                localGridPtr[i * BLOCK_SIZE + j] = mainGrid[(b * BLOCK_SIZE*BLOCK_SIZE) + (i * BLOCK_SIZE + j)];

                //std::cout << "[" << i * BLOCK_SIZE + j << "]" << "[" << globalI * N + globalJ << "]" << std::endl;
                //std::cout <<"b = " << b <<"; i = " << i << "; j = " << j <<  "; i * BLOCK_SIZE + j = " << i * BLOCK_SIZE + j
                // <<  "; (b+1) * i * BLOCK_SIZE + j = " << (b*BLOCK_SIZE*BLOCK_SIZE) + (i * BLOCK_SIZE + j) << std::endl;  
            }
        }

        hostBlocks[b] = BlockOfGrid(xMin, xMax, yMin, yMax, BLOCK_SIZE, localGridPtr);
    }

    //comparison of localGrid and mainGrid
    /*std::cout << "Befor computation" << std::endl;
    for (int i = 0; i < 100 * 100; ++i){
        std::cout << "Grid ["<< i <<"] " << hostLocalGrids[i] << "  MainGrid [" << i <<"] " << mainGrid[i] << std::endl;
    }*/

    /*for (int i = 0; i < NUM_BLOCKS; ++i){
        std::cout << "xMin: " << hostBlocks[i].xMin << ", yMin: " << hostBlocks[i].yMin
         <<  ", xMax: " << hostBlocks[i].xMax <<  ", yMax: " << hostBlocks[i].yMax << std::endl;
    }*/

    // Allocate memory on device
    float* deviceLocalGrids;
    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMemcpy(deviceLocalGrids, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice); //Check the syntax

    BlockOfGrid* deviceBlocks;
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        hostBlocks[b].localGrid = deviceLocalGrids + b * BLOCK_SIZE * BLOCK_SIZE;
    }
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS);
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    for (int i = 0; i < 5; ++i){
        // Launch kernel
        processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, NUM_BLOCKS, mainGrid);
        cudaDeviceSynchronize();

        // Copy back result
        cudaMemcpy(hostLocalGrids, deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);

        float* d_input;
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
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // 3. Run max reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output,
                            NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);

        // 4. Copy result to host
        float maxValue;
        cudaMemcpy(&maxValue, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "\n Max grid value across all blocks: " << maxValue << "\n";

        // Cleanup CUB allocations
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp_storage);

        // Copy back results to mainGrid for visualization
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            //int xMin = hostBlocks[b].xMin;
            //int yMin = hostBlocks[b].yMin;
            float* localGridPtr = &hostLocalGrids[b * BLOCK_SIZE * BLOCK_SIZE];

            for (int i = 0; i < BLOCK_SIZE; ++i) {
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    //int globalI = xMin + i;
                    //int globalJ = yMin + j;
                    mainGrid[(b * BLOCK_SIZE*BLOCK_SIZE) + (i * BLOCK_SIZE + j)] = localGridPtr[i * BLOCK_SIZE + j];
                }
            }
        }

        // Print sample 
        /*for (int i = 0; i < NUM_BLOCKS; ++i) {
            std::cout << "Block " << i << " sample (10,10): "
                    << hostLocalGrids[i * BLOCK_SIZE * BLOCK_SIZE + 10 * BLOCK_SIZE + 10] << "\n";
        }*/

        //After computing
        /*for (int i = 0; i < 100 * 100; ++i){
            std::cout << "Grid ["<< i <<"] " << hostLocalGrids[i] << std::endl;
        }*/
    }

    // Displays the main grid
    /*for (int i = 0; i < 100 * 100; ++i){
        std::cout << "mainGrid ["<< i <<"] " << mainGrid[i] << std::endl;
    }
    std::cout << "--------------------------------------------" << std::endl;*/

    // Cleanup
    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    delete[] hostLocalGrids;
    delete[] mainGrid;

    return 0;
}
