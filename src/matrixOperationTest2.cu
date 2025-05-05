#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define N 150  // Global grid size
#define NUM_BLOCKS 9  // Number of blocks
#define BLOCK_SIZE 50  // Size of each block (50x50)
#define ROWS 200
#define COLS 200
#define EPS 1e-3

struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, width;
    float* localGrid;  // Points to a subregion in the global grid

    //Physical constants
    float alpha = 0.35;
    float dx = 1.0;
    float dy = 1.0;
    float dt = 0.1;
    float r_x = alpha * dt/(2 * dx * dx);
    float r_y = alpha * dt/(2 * dy * dy);
    float tempDiff = 100.0;
    float maxTempDiff = 0.0;

    __host__ __device__
    BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, int gridWidth = 0, float* gridPtr = nullptr)
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), width(gridWidth), localGrid(gridPtr) {}

    // Utility functions
    __device__ float& at(int i, int j) {
        return localGrid[i * width + j];
    }

    __device__ inline float d_max(float a, float b) {
        return a > b ? a : b;
    }   

    __device__ inline float d_min(float a, float b) {
        return a < b ? a : b;
    }

    __device__ float compute(float* dGrid) {

        for (int i = 0; i < (xMax - xMin); ++i) {
            for (int j = 0; j < (yMax - yMin); ++j) {
                //localGrid[i * width + j] = dGrid[(xMin + i) * N + (yMin + j)] + 5.0f;
                tempDiff = r_x * (
                    dGrid[(xMin + i + 1) * N + (yMin + j)] -
                    2 * dGrid[(xMin + i) * N + (yMin + j)] +
                    dGrid[(xMin + i - 1) * N + (yMin + j)]
                ) +
                r_y * (
                    dGrid[(xMin + i) * N + (yMin + j + 1)] -
                    2 * dGrid[(xMin + i) * N + (yMin + j)] +
                    dGrid[(xMin + i) * N + (yMin + j - 1)]
                );
                localGrid[i * width + j] = dGrid[(xMin + i) * N + (yMin + j)] + tempDiff;
                //maxTempDiff = d_max(maxTempDiff, tempDiff);
            }
        }
        //maxTempDiff = d_max(maxTempDiff, tempDiff);
        //printf("Max Temp difference is %f \n", maxTempDiff);

        if (xMin > 0 && xMax < ROWS && yMin > 0 && yMax < COLS) {
            return;
        }

        // Set edge cells to 0 only if the block touches a boundary
        if (xMin == 0) {  // Top boundary
            for (int j = yMin; j < yMax; ++j) {
                localGrid[(0) * width + (j - yMin)] = 0.0f;
            }
        }
        if (xMax == ROWS) {  // Bottom boundary
            for (int j = yMin; j < yMax; ++j) {
                localGrid[(xMax - xMin - 1) * width + (j - yMin)] = 0.0f;
            }
        }
        if (yMin == 0) {  // Left boundary
            for (int i = xMin; i < xMax; ++i) {
                localGrid[(i - xMin) * width + (0)] = 0.0f;
            }
        }
        if (yMax == COLS) {  // Right boundary
            for (int i = xMin; i < xMax; ++i) {
                localGrid[(i - xMin) * width + (yMax - yMin - 1)] = 0.0f;
            }
        }
        return maxTempDiff;
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks, float* Grid, float epsilon) {
    __shared__ float sharedMax[NUM_BLOCKS];  // Or use blockDim.x if flexible
    int idx = threadIdx.x;

    float localMaxTemp = 0.0f;
    float globalMaxTemp = 0.0f;
    int i = 0;

    do {
        if (idx < numBlocks) {
            // 1. Run compute and get local max temp delta
            localMaxTemp = blocks[idx].compute(Grid);

            // 2. Write back localGrid to global Grid
            BlockOfGrid& block = blocks[idx];
            for (int i = block.xMin; i < block.xMax; ++i) {
                for (int j = block.yMin; j < block.yMax; ++j) {
                    Grid[i * N + j] = block.localGrid[(i - block.xMin) * block.width + (j - block.yMin)];
                }
            }

            sharedMax[idx] = localMaxTemp;
        } else {
            sharedMax[idx] = 0.0f;
        }

        __syncthreads();

        // 3. In-place reduction to find max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (idx < stride) {
                sharedMax[idx] = max(sharedMax[idx], sharedMax[idx + stride]);
            }
            __syncthreads();
        }

        globalMaxTemp = sharedMax[0];
        __syncthreads();
        ++i;
        //printf("%f \n",globalMaxTemp);

    } while (i < 1);
}

int main() {
    float* mainGrid = new float[N * N];
    float* hostLocalGrids = new float[NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE];

    // Initialize mainGrid
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0 || i == N - 1) {
                mainGrid[(i * N) + j] = 0.0f;
            } else {
                mainGrid[(i * N) + j] = 0.0f;
            }
        }
    }
    //mainGrid[(50 * 100 + 50)] = 205.0f;
    /*for (int i = 60; i < 80; ++i){
        for (int j = 60; j <80; ++j){
            mainGrid[(i * N) + j] = 100.0f;
        }
    }*/

    BlockOfGrid hostBlocks[NUM_BLOCKS];
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        //std::cout << "b = " << b << std::endl;
        //std::cout << "b/2 = " << (b/2) << " Block size = " << BLOCK_SIZE << std::endl; 
        int xMin = (b / 3) * BLOCK_SIZE;
        //std::cout << "xMin = " << xMin << std::endl;
        int xMax = xMin + BLOCK_SIZE;
        //std::cout << "xMax = " << xMax << std::endl;
        int yMin = (b % 3) * BLOCK_SIZE;
        //std::cout << "yMin = " << yMin << std::endl;
        int yMax = yMin + BLOCK_SIZE;
        //std::cout << "yMax = " << yMax << std::endl;
        //std::cout << "==================================== \n";
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

    /*for (int i = 0; i < NUM_BLOCKS; ++i){
        std::cout << "xMin: " << hostBlocks[i].xMin << ", yMin: " << hostBlocks[i].yMin
         <<  ", xMax: " << hostBlocks[i].xMax <<  ", yMax: " << hostBlocks[i].yMax << std::endl;
    }*/

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

    float* d_output;
    cudaMalloc(&d_output, sizeof(float));

    // Temporary storage
    float* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temp storage size
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, deviceMainGrid, d_output,
                        NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for (int i = 0; i < 1; ++i){
        // Launch kernel
        processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, NUM_BLOCKS, deviceMainGrid, EPS);
        cudaDeviceSynchronize();
        
        /*cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, deviceMainGrid, d_output,
                            NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE);


        float maxValue;
        cudaMemcpy(&maxValue, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "\n Max grid value across all blocks: " << maxValue << "\n";*/
        cudaMemcpy(mainGrid, deviceMainGrid, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        //After computing
        std::cout << "i," << "j," << "value\n";
        for (int l_1 = 0; l_1 < N; ++l_1){
            for (int l_2 = 0; l_2 < N; ++l_2){
                std::cout << l_1 << "," << l_2<< "," << mainGrid[l_1 * N + l_2] << std::endl;
            }
        }
    }

    // Copy back result
    cudaMemcpy(hostLocalGrids, deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(mainGrid, deviceMainGrid, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    cudaFree(deviceMainGrid);
    cudaFree(d_output);

    cudaFree(deviceBlocks);
    cudaFree(d_temp_storage);

    delete[] hostLocalGrids;
    delete[] mainGrid;

    return 0;
}