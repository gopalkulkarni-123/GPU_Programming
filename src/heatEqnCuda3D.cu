#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define N 9  // Global grid size
#define NUM_BLOCKS 27  // Number of blocks
#define BLOCK_SIZE 3  // Size of each block 
#define ROWS 9
#define COLS 9
#define DEPTH 9
#define EPS 1e-6

struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, zMin, zMax;
    int blockLength, blockBreadth, blockDepth;
    float* localGrid;  // Points to a subregion in the global grid

    //Physical constants
    float alpha = 0.35;
    float dx = 1.0;
    float dy = 1.0;
    float dz = 1.0;
    float dt = 0.1;
    float r_x = alpha * dt/(2 * dx * dx);
    float r_y = alpha * dt/(2 * dy * dy);
    float r_z =  alpha * dt/(2 * dz * dz);
    float tempDiff = 0.0;
    float maxTempDiff = 0.0;
    unsigned long long time = 0;


    __host__ __device__
    BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, int z_min = 0, int z_max = 0, float* gridPtr = nullptr) // add zmin, zmax etc and take care of gridwidth and etc. 
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), zMin(z_min), zMax(z_max), localGrid(gridPtr), 
        blockBreadth(y_max - y_min), blockLength(x_max - x_min), blockDepth(z_max - z_min) {}

    // Utility functions

    __device__ inline float d_max(float a, float b) {
        return a > b ? a : b;
    }   

    __device__ inline float d_min(float a, float b) {
        return a < b ? a : b;
    }

    __device__ float compute(float* dGrid) {
        //unsigned long long start = clock64();

        for (int z = 0; z < blockDepth; ++z) {
            for (int y = 0; y < blockBreadth; ++y) { 
                for (int x = 0; x < blockLength; ++x) {

                    tempDiff = r_x * (
                        dGrid[(xMin + x + 1) * (N*N) + (yMin + y) * N + (zMin + z)] -
                        2 * dGrid[(xMin + x) * (N*N) + (yMin + y) * N + (zMin + z)] +
                        dGrid[(xMin + x - 1) * (N*N) + (yMin + y) * N + (zMin + z)]
                    ) +
                    r_y * (
                        dGrid[(xMin + x) * (N*N) + (yMin + y + 1) * N + (zMin + z)] -
                        2 * dGrid[(xMin + x) * (N*N) + (yMin + y) * N + (zMin + z)] +
                        dGrid[(xMin + x) * (N*N) + (yMin + y - 1) * N + (zMin + z)]
                    ) +
                    r_z * (
                        dGrid[(xMin + x) * (N*N) + (yMin + y) * N + (zMin + z + 1)] -
                        2 * dGrid[(xMin + x) * (N*N) + (yMin + y) * N + (zMin + z)] +
                        dGrid[(xMin + x) * (N*N) + (yMin + y) * N + (zMin + z - 1)]
                    );

                    localGrid[(z * ROWS * COLS) + y*ROWS + x] = dGrid[((xMin + x) * (N*N)) + ((yMin + y) * N) + (zMin + z)] + tempDiff; // index = k * (height * width) + y * width + i;
                    maxTempDiff = d_max(maxTempDiff, tempDiff);
                }
            }
        }
        //maxTempDiff = d_max(maxTempDiff, tempDiff);
        //printf("Max Temp difference is %f \n", maxTempDiff);

        if (xMin > 0 && xMax < ROWS && yMin > 0 && yMax < COLS && zMin > 0 && zMax < DEPTH) {
            //unsigned long long end = clock64();
            //time = time + (end - start);
            return maxTempDiff;
        }

        // Set edge cells to 0 only if the block touches a boundary
        if (xMin == 0) {  // Left Boundary
            for (int z = zMin; z < zMax; ++z) {
                for (int y = yMin; y < yMax; ++y){
                    localGrid[(((z - zMin) * (blockBreadth * blockLength)) + ((y - yMin)*(blockLength)) + (0))] = 100.0f;
                    //printf("Left Boundary index = %d \n", ((z - zMin) * (blockBreadth * blockLength)) + ((y - yMin)*(blockLength)) + (0));
                    //printf("z = %d; (z - zMin) = %d; index = %d \n", z, (z - zMin), (((z - zMin) * (ROWS * COLS)) + ((y - yMin)*(ROWS)) + (0)));
                    //printf("Left Boundary index = %d \n", ((z - zMin) * (ROWS * COLS)) + ((y - yMin)*(ROWS)) + (0));
                    //printf("yMin = %d; yMax = %d \n", yMin, yMax);    
                }
            }
        }

        if (xMax == ROWS) {  // Right Boundary
            for (int z = zMin; z < zMax; ++z) {
                for (int y = yMin; y < yMax; ++y){
                    localGrid[(((z - zMin) * (blockBreadth * blockLength)) + ((y - yMin)*(blockLength)) + (xMax - 1 - xMin))] = 200.0f;
                    //printf("Right Boundary index = %d \n", ((z - zMin) * (blockBreadth * blockLength)) + ((y - yMin)*(blockLength)) + (xMax - 1 - xMin));
                }
            }
        }

        if (yMin == 0) {  // Bottom boundary
            for (int z = zMin; z < zMax; ++z) {
                for (int x = xMin; x < xMax; ++x){
                    localGrid[(z - zMin) * (blockBreadth * blockLength) + (0)*(ROWS) + (x - xMin)] = 300.0f;
                    //printf("Bottom Boundary index = %d \n", ((z - zMin) * (blockBreadth * blockLength) + (0)*(blockLength) + (x - xMin)));
                }
            }
        }

        if (yMax == COLS) {  // Top boundary
            for (int z = zMin; z < zMax; ++z) {
                for (int x = xMin; x < xMax; ++x){
                    localGrid[(z - zMin) * (blockBreadth * blockLength) + (yMax - 1 - yMin)*(blockLength) + (x - xMin)] = 400.0f;
                    //printf("Top Boundary index = %d \n", ((z - zMin) * (blockBreadth * blockLength) + (yMax - 1 - yMin)*(blockLength) + (x - xMin)));
                    //printf("zMin = %d || z = %d || zMax = %d \n", zMin, z, zMax);
                }
            }
        }

        if (zMin == 0){
            for (int y = yMin; y < yMax; ++y){
                for (int x = xMin; x < xMax; ++x){
                    localGrid[(0) * (blockBreadth * blockLength) + (y - yMin)*(blockLength) + (x - xMin)] = 500.0f;
                    //printf("Front Boundary index = %d \n", ((0) * (blockBreadth * blockLength) + (y - yMin)*(blockLength) + (x - xMin)));
                    //printf("xMin = %d || xMax = %d \n yMin = %d || yMax = %d \n zMin = %d || zMax = %d \n -------- \n", xMin, xMax, yMin, yMax, zMin, zMax);
                }
            }
        }

        if (zMax == DEPTH){
            for (int y = yMin; y < yMax; ++y){
                for (int x = xMin; x < xMax; ++x){
                    localGrid[(zMax - 1 - zMin) * (blockBreadth * blockLength) + (y - yMin)*(blockLength) + (x - xMin)] = 600.0f;
                    //printf("Back Boundary index = %d \n", ((zMax - 1 - zMin) * (blockBreadth * blockLength) + (y - yMin)*(blockLength) + (x - xMin)));
                    //printf("xMin = %d || xMax = %d \n yMin = %d || yMax = %d \n zMin = %d || zMax = %d \n -------- \n", xMin, xMax, yMin, yMax, zMin, zMax);
                }
            }
        }
    

        //unsigned long long end = clock64();
        //time = time + (end - start);
    return maxTempDiff;
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks, float* Grid, float epsilon) {
    __shared__ float sharedMax[NUM_BLOCKS];  // Or use blockDim.x if flexible
    int idx = threadIdx.x;

    float localMaxTemp = 100.0f;
    //float globalMaxTemp = 0.0f;
    float tempValues[2];
    int count = 0;

    do {
        tempValues[0] = tempValues[1];
        if (idx < numBlocks) {
            // 1. Run compute and get local max temp delta
            localMaxTemp = blocks[idx].compute(Grid);
            //printf("Max temp of my block is %f \n", localMaxTemp);

            // 2. Write back localGrid to global Grid
            BlockOfGrid& block = blocks[idx];
            for (int z = block.zMin; z < block.zMax; ++z) {
                for (int y = block.yMin; y < block.yMax; ++y) {
                    for(int x = block.xMin; x < block.xMax; ++x){
                        Grid[(z * (N*N)) + (y*N) + x] = block.localGrid[(z - block.zMin) * (block.blockBreadth*block.blockLength) 
                                                    + (y - block.yMin)*(block.blockLength) 
                                                    + (x - block.xMin)];
                        /*if (block.xMin == 0){                        
                            printf("Id = %d || Grid index = %d ||  local index = %d \n",(idx), ((z * (N*N)) + y*N + x),  ((z - block.zMin) * (ROWS*COLS) 
                                                        + (y - block.yMin)*(ROWS) 
                                                        + (x - block.xMin)));}*/
                    }
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

        tempValues[1] = sharedMax[0];
        __syncthreads();
        
        //printf("Idx is %d and Global max temp is %f < %f \n", idx, globalMaxTemp, epsilon);
        /*if(idx == 0){
            printf("Max temperature difference is %f \n", abs(tempValues[1] - tempValues[0]));
        }*/
        ++count;

    } while(count < 1);
    //while ((abs(tempValues[1] - tempValues[0])) > epsilon || i < 10) ;
    /*for (int i_1 = 0; i_1 < NUM_BLOCKS; ++i_1){
            printf("Time for block[%d] is %llu with xMin = %d; xMax = %d; yMin = %d; yMax = %d \n", i_1, blocks[i_1].time, blocks[i_1].xMin, blocks[i_1].xMax, blocks[i_1].yMin, blocks[i_1].yMax);
        }*/
        
}

int main() {
    float* mainGrid = new float[N * N * N];
    float* hostLocalGrids = new float[NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

    // Initialize mainGrid
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) { // Add the z direction iteration
                if (x == 0 || x == N - 1) {
                    mainGrid[(z * (N*N) + y*N + x)] = 0.0f;
                } else {
                    mainGrid[(z * (N*N) + y*N + x)] = 0.0f;
                }
            }
        }
    }

    BlockOfGrid hostBlocks[NUM_BLOCKS];
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        //std::cout << "b = " << b << std::endl;
        //std::cout << "b/2 = " << (b/2) << " Block size = " << BLOCK_SIZE << std::endl; 
        int xMin = (b * BLOCK_SIZE) % (BLOCK_SIZE*BLOCK_SIZE);
        //int xMin = ((b / BLOCK_SIZE) * BLOCK_SIZE) % (BLOCK_SIZE * BLOCK_SIZE);
        //std::cout << "xMin = " << xMin << std::endl;
        int xMax = xMin + BLOCK_SIZE;
        //std::cout << "xMax = " << xMax << std::endl;
        //int yMin = (b * BLOCK_SIZE) % (BLOCK_SIZE*BLOCK_SIZE);
        int yMin = ((b / BLOCK_SIZE) * BLOCK_SIZE) % (BLOCK_SIZE * BLOCK_SIZE);
        //std::cout << "yMin = " << yMin << std::endl;
        int yMax = yMin + BLOCK_SIZE;
        //std::cout << "yMax = " << yMax << std::endl;
        int zMin = (b / (BLOCK_SIZE * BLOCK_SIZE)) * BLOCK_SIZE;
        //std::cout << "zMin = " << zMin << std::endl;
        int zMax = zMin + BLOCK_SIZE;
        //std::cout << "zMax = " << zMax << std::endl;
        //std::cout << "==================================== \n";

        //add zMin and zMax

        //int width = yMax - yMin;
        float* localGridPtr = &hostLocalGrids[b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        // Copy corresponding block from mainGrid
        for (int z = 0; z < BLOCK_SIZE; ++z) {
            for (int y = 0; y < BLOCK_SIZE; ++y) {
                for (int x = 0; x < BLOCK_SIZE; ++x){
                    localGridPtr[((z * BLOCK_SIZE * BLOCK_SIZE) + (y*BLOCK_SIZE) + x)] = mainGrid[((b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + ((z * BLOCK_SIZE * BLOCK_SIZE) + (y * BLOCK_SIZE) + x))];
                    //std::cout << "Local grid : " << ((z * BLOCK_SIZE * BLOCK_SIZE) + (y*BLOCK_SIZE) + x) << "||" << "Main Grid : " << ((b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) + ((z * BLOCK_SIZE * BLOCK_SIZE) + (y * BLOCK_SIZE) + x)) << std::endl;
                }
            }
        }

        hostBlocks[b] = BlockOfGrid(xMin, xMax, yMin, yMax, zMin, zMax, localGridPtr);
    }

    
    /*for (int i = 0; i < NUM_BLOCKS; ++i){
        if (hostBlocks[i].xMax == ROWS){
            std::cout << "xMin: " << hostBlocks[i].xMin << ", xMax: " << hostBlocks[i].xMax
         <<  ", yMin: " << hostBlocks[i].yMin <<  ", yMax: " << hostBlocks[i].yMax
         <<  ", zMin: " << hostBlocks[i].zMin <<  ", zMax: " << hostBlocks[i].zMax <<std::endl;
        }
    }*/

    // Allocate memory on device
    float* deviceLocalGrids;
    float* deviceMainGrid;

    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE); // Increase the size to accomodate the z direction
    cudaMalloc(&deviceMainGrid, sizeof(float) * N * N * N); // Increase the size to accomodate the z direction
    
    cudaMemcpy(deviceLocalGrids, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice); //Check the syntax
    cudaMemcpy(deviceMainGrid, mainGrid, sizeof(float) * N * N * N, cudaMemcpyHostToDevice);

    BlockOfGrid* deviceBlocks;
    for (int b = 0; b < NUM_BLOCKS; ++b) {

        hostBlocks[b].localGrid = deviceLocalGrids + (b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
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
                        NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
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
        cudaMemcpy(mainGrid, deviceMainGrid, (sizeof(float) * N * N * N), cudaMemcpyDeviceToHost);
        //After computing
        std::cout << "x," << "y," << "z," << "value\n";
        for (int z = 0; z < N; ++z){
            for (int y = 0; y < N; ++y){
                for (int x = 0; x < N; ++x){
                    std::cout << x << "," << y << "," << z << "," << mainGrid[(z * N * N) + y*N + x] << std::endl;
                }
            }
        }
    }

    // Copy back result
    cudaMemcpy(hostLocalGrids, deviceLocalGrids, (sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), cudaMemcpyDeviceToHost);
    cudaMemcpy(mainGrid, deviceMainGrid, (sizeof(float) * N * N * N), cudaMemcpyDeviceToHost);

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