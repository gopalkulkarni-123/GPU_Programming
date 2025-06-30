#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define N 150  // Global grid size
#define NUM_BLOCKS 27  // Number of blocks (3x3x3)
#define BLOCK_SIZE 50  // Size of each block 
#define ROWS 150
#define COLS 150
#define DEPTH 150
#define EPS 1e-6
#define MAX_ITERATIONS 1000

struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, zMin, zMax;
    int blockLength, blockBreadth, blockDepth;
    float* localGrid;  // Points to a subregion in the global grid
    
    // Boundary flags for optimization
    bool isOnXMinBoundary, isOnXMaxBoundary;
    bool isOnYMinBoundary, isOnYMaxBoundary;
    bool isOnZMinBoundary, isOnZMaxBoundary;
    bool isInteriorBlock;  // True if no boundaries at all
    
    // Physical constants
    float alpha = 0.35f;
    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    float dt = 0.1f;
    float r_x = alpha * dt / (dx * dx);
    float r_y = alpha * dt / (dy * dy);
    float r_z = alpha * dt / (dz * dz);
    
    __host__ __device__
    BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, 
                int z_min = 0, int z_max = 0, float* gridPtr = nullptr) 
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), zMin(z_min), zMax(z_max), 
          localGrid(gridPtr), blockBreadth(y_max - y_min), 
          blockLength(x_max - x_min), blockDepth(z_max - z_min) {
        
        // Determine which boundaries this block touches
        isOnXMinBoundary = (xMin == 0);
        isOnXMaxBoundary = (xMax == N);
        isOnYMinBoundary = (yMin == 0);
        isOnYMaxBoundary = (yMax == N);
        isOnZMinBoundary = (zMin == 0);
        isOnZMaxBoundary = (zMax == N);
        
        // Check if this is a completely interior block
        isInteriorBlock = !isOnXMinBoundary && !isOnXMaxBoundary && 
                         !isOnYMinBoundary && !isOnYMaxBoundary && 
                         !isOnZMinBoundary && !isOnZMaxBoundary;
    }
    
    __device__ inline float d_max(float a, float b) {
        return a > b ? a : b;
    }   
    
    __device__ float compute(float* dGrid) {
        float maxTempDiff = 0.0f;
        
        // First, copy current values to local grid
        for (int z = 0; z < blockDepth; ++z) {
            for (int y = 0; y < blockBreadth; ++y) { 
                for (int x = 0; x < blockLength; ++x) {
                    int globalX = xMin + x;
                    int globalY = yMin + y;
                    int globalZ = zMin + z;
                    int globalIdx = (globalZ * N * N) + (globalY * N) + globalX;
                    int localIdx = (z * blockBreadth * blockLength) + (y * blockLength) + x;
                    localGrid[localIdx] = dGrid[globalIdx];
                }
            }
        }
        
        // Apply heat equation - optimized based on block type
        if (isInteriorBlock) {
            // Fast path for interior blocks - no boundary checks needed
            maxTempDiff = computeInteriorOnly(dGrid);
        } else {
            // Slower path for boundary blocks - need to check each point
            maxTempDiff = computeWithBoundaryChecks(dGrid);
        }
        
        // Apply boundary conditions only if this block touches boundaries
        if (!isInteriorBlock) {
            applyBoundaryConditions();
        }
        
        return maxTempDiff;
    }
    
    __device__ float computeInteriorOnly(float* dGrid) {
        float maxTempDiff = 0.0f;
        
        // All points in this block are interior, so no boundary checks needed
        for (int z = 0; z < blockDepth; ++z) {
            for (int y = 0; y < blockBreadth; ++y) { 
                for (int x = 0; x < blockLength; ++x) {
                    int globalX = xMin + x;
                    int globalY = yMin + y;
                    int globalZ = zMin + z;
                    
                    // Apply finite difference heat equation
                    float center = dGrid[(globalZ * N * N) + (globalY * N) + globalX];
                    
                    float left = dGrid[(globalZ * N * N) + (globalY * N) + (globalX - 1)];
                    float right = dGrid[(globalZ * N * N) + (globalY * N) + (globalX + 1)];
                    float bottom = dGrid[(globalZ * N * N) + ((globalY - 1) * N) + globalX];
                    float top = dGrid[(globalZ * N * N) + ((globalY + 1) * N) + globalX];
                    float front = dGrid[((globalZ - 1) * N * N) + (globalY * N) + globalX];
                    float back = dGrid[((globalZ + 1) * N * N) + (globalY * N) + globalX];
                    
                    float tempDiff = r_x * (left - 2.0f * center + right) +
                                    r_y * (bottom - 2.0f * center + top) +
                                    r_z * (front - 2.0f * center + back);
                    float newValue = center + tempDiff;
                    int localIdx = (z * blockBreadth * blockLength) + (y * blockLength) + x;
                    localGrid[localIdx] = newValue;
                    maxTempDiff = d_max(maxTempDiff, fabsf(tempDiff));
                }
            }
        }
        return maxTempDiff;
    }
    
    __device__ float computeWithBoundaryChecks(float* dGrid) {
        float maxTempDiff = 0.0f;
        
        // Need to check each point for boundary conditions
        for (int z = 0; z < blockDepth; ++z) {
            for (int y = 0; y < blockBreadth; ++y) { 
                for (int x = 0; x < blockLength; ++x) {
                    int globalX = xMin + x;
                    int globalY = yMin + y;
                    int globalZ = zMin + z;
                    
                    // Skip if this is a boundary point
                    if (globalX == 0 || globalX == N-1 || 
                        globalY == 0 || globalY == N-1 || 
                        globalZ == 0 || globalZ == N-1) {
                        continue;
                    }
                    
                    // Apply finite difference heat equation
                    float center = dGrid[(globalZ * N * N) + (globalY * N) + globalX];
                    
                    float left = dGrid[(globalZ * N * N) + (globalY * N) + (globalX - 1)];
                    float right = dGrid[(globalZ * N * N) + (globalY * N) + (globalX + 1)];
                    float bottom = dGrid[(globalZ * N * N) + ((globalY - 1) * N) + globalX];
                    float top = dGrid[(globalZ * N * N) + ((globalY + 1) * N) + globalX];
                    float front = dGrid[((globalZ - 1) * N * N) + (globalY * N) + globalX];
                    float back = dGrid[((globalZ + 1) * N * N) + (globalY * N) + globalX];
                    
                    float tempDiff = r_x * (left - 2.0f * center + right) +
                                    r_y * (bottom - 2.0f * center + top) +
                                    r_z * (front - 2.0f * center + back);
                    float newValue = center + tempDiff;
                    int localIdx = (z * blockBreadth * blockLength) + (y * blockLength) + x;
                    localGrid[localIdx] = newValue;
                    maxTempDiff = d_max(maxTempDiff, fabsf(tempDiff));
                }
            }
        }
        return maxTempDiff;
    }
    
    __device__ void applyBoundaryConditions() {
        // Only apply boundary conditions to points that are actually on boundaries
        for (int z = 0; z < blockDepth; ++z) {
            for (int y = 0; y < blockBreadth; ++y) { 
                for (int x = 0; x < blockLength; ++x) {
                    int globalX = xMin + x;
                    int globalY = yMin + y;
                    int globalZ = zMin + z;
                    int localIdx = (z * blockBreadth * blockLength) + (y * blockLength) + x;
                    
                    // Check if this point is on a boundary
                    bool onBoundary = false;
                    float boundaryValue = 0.0f;
                    
                    // Fixed temperature boundaries
                    if ((isOnXMinBoundary && x == 0) || 
                        (isOnXMaxBoundary && x == blockLength - 1) ||
                        (isOnYMinBoundary && y == 0) || 
                        (isOnYMaxBoundary && y == blockBreadth - 1)) {
                        onBoundary = true;
                        boundaryValue = 0.0f;  // Cold boundaries
                    }
                    else if ((isOnZMinBoundary && z == 0) || 
                             (isOnZMaxBoundary && z == blockDepth - 1)) {
                        onBoundary = true;
                        boundaryValue = 100.0f;  // Hot boundaries (heat sources)
                    }
                    
                    if (onBoundary) {
                        localGrid[localIdx] = boundaryValue;
                    }
                }
            }
        }
    }
};

__global__ void processBlocks(BlockOfGrid* blocks, int numBlocks, float* Grid, float epsilon, int* converged) {
    __shared__ float sharedMax[NUM_BLOCKS];
    int idx = threadIdx.x;

    float localMaxTemp = 0.0f;
    
    if (idx < numBlocks) {
        // 1. Run compute and get local max temp delta
        localMaxTemp = blocks[idx].compute(Grid);
        sharedMax[idx] = localMaxTemp;
    } else {
        sharedMax[idx] = 0.0f;
    }

    __syncthreads();

    // 2. Reduction to find global maximum temperature difference
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            sharedMax[idx] = max(sharedMax[idx], sharedMax[idx + stride]);
        }
        __syncthreads();
    }

    __syncthreads();  // Ensure reduction is complete

    // 3. Write back localGrid to global Grid
    if (idx < numBlocks) {
        BlockOfGrid& block = blocks[idx];
        for (int z = block.zMin; z < block.zMax; ++z) {
            for (int y = block.yMin; y < block.yMax; ++y) {
                for (int x = block.xMin; x < block.xMax; ++x) {
                    int globalIdx = (z * N * N) + (y * N) + x;
                    int localIdx = ((z - block.zMin) * block.blockBreadth * block.blockLength) + 
                                   ((y - block.yMin) * block.blockLength) + 
                                   (x - block.xMin);
                    Grid[globalIdx] = block.localGrid[localIdx];
                }
            }
        }
    }

    // 4. Check convergence
    if (idx == 0) {
        if (sharedMax[0] < epsilon) {
            *converged = 1;
        }
    }
}

int main() {
    float* mainGrid = new float[N * N * N];
    float* hostLocalGrids = new float[NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

    // Initialize mainGrid with proper physics
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int idx = (z * N * N) + (y * N) + x;
                
                // Boundary conditions
                if (x == 0 || x == N-1 || y == 0 || y == N-1) {
                    mainGrid[idx] = 0.0f;  // Cold side boundaries
                } else if (z == 0 || z == N-1) {
                    mainGrid[idx] = 100.0f;  // Hot front/back boundaries
                } else {
                    mainGrid[idx] = 20.0f;  // Initial interior temperature
                }
            }
        }
    }

    // Create blocks with corrected indexing
    BlockOfGrid hostBlocks[NUM_BLOCKS];
    int blocks_per_row = N / BLOCK_SIZE;  // 3
    int blocks_per_slice = blocks_per_row * blocks_per_row;  // 9
    
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        int block_z = b / blocks_per_slice;
        int block_y = (b % blocks_per_slice) / blocks_per_row;
        int block_x = b % blocks_per_row;
        
        int xMin = block_x * BLOCK_SIZE;
        int xMax = xMin + BLOCK_SIZE;
        int yMin = block_y * BLOCK_SIZE;
        int yMax = yMin + BLOCK_SIZE;
        int zMin = block_z * BLOCK_SIZE;
        int zMax = zMin + BLOCK_SIZE;
        
        float* localGridPtr = &hostLocalGrids[b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        // Copy corresponding block from mainGrid to localGrid
        for (int z = 0; z < BLOCK_SIZE; ++z) {
            for (int y = 0; y < BLOCK_SIZE; ++y) {
                for (int x = 0; x < BLOCK_SIZE; ++x) {
                    int globalIdx = ((zMin + z) * N * N) + ((yMin + y) * N) + (xMin + x);
                    int localIdx = (z * BLOCK_SIZE * BLOCK_SIZE) + (y * BLOCK_SIZE) + x;
                    localGridPtr[localIdx] = mainGrid[globalIdx];
                }
            }
        }

        hostBlocks[b] = BlockOfGrid(xMin, xMax, yMin, yMax, zMin, zMax, localGridPtr);
    }

    // Allocate device memory
    float* deviceLocalGrids;
    float* deviceMainGrid;
    int* deviceConverged;
    int hostConverged = 0;

    cudaMalloc(&deviceLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
    cudaMalloc(&deviceMainGrid, sizeof(float) * N * N * N);
    cudaMalloc(&deviceConverged, sizeof(int));
    
    cudaMemcpy(deviceLocalGrids, hostLocalGrids, sizeof(float) * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMainGrid, mainGrid, sizeof(float) * N * N * N, cudaMemcpyHostToDevice);

    // Update host blocks to point to device memory
    BlockOfGrid* deviceBlocks;
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        hostBlocks[b].localGrid = deviceLocalGrids + (b * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
    }
    
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS);
    cudaMemcpy(deviceBlocks, hostBlocks, sizeof(BlockOfGrid) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    // Simulation loop
    int iteration = 0;
    std::cout << "Starting heat diffusion simulation..." << std::endl;
    std::cout << "Hot sources: z=0 and z=8 planes (100°C)" << std::endl;
    std::cout << "Cold boundaries: x=0, x=8, y=0, y=8 edges (0°C)" << std::endl;
    std::cout << "Initial interior: 20°C" << std::endl;
    
    do {
        hostConverged = 0;
        cudaMemcpy(deviceConverged, &hostConverged, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch kernel
        processBlocks<<<1, NUM_BLOCKS>>>(deviceBlocks, NUM_BLOCKS, deviceMainGrid, EPS, deviceConverged);
        cudaDeviceSynchronize();
        
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            break;
        }
        
        cudaMemcpy(&hostConverged, deviceConverged, sizeof(int), cudaMemcpyDeviceToHost);
        iteration++;
        
        if (iteration % 100 == 0) {
            cudaMemcpy(mainGrid, deviceMainGrid, sizeof(float) * N * N * N, cudaMemcpyDeviceToHost);
            float centerTemp = mainGrid[(4 * N * N) + (4 * N) + 4];  // Center point
            std::cout << "Iteration " << iteration << ", Center temperature: " << centerTemp << "°C" << std::endl;
        }
        
    } while (!hostConverged && iteration < MAX_ITERATIONS);

    std::cout << "Simulation completed after " << iteration << " iterations" << std::endl;
    if (hostConverged) {
        std::cout << "Converged to epsilon = " << EPS << std::endl;
    } else {
        std::cout << "Maximum iterations reached" << std::endl;
    }

    // Copy results back
    cudaMemcpy(mainGrid, deviceMainGrid, sizeof(float) * N * N * N, cudaMemcpyDeviceToHost);

    // Output some key temperatures for validation
    std::cout << "\nKey temperature points:" << std::endl;
    std::cout << "Center (4,4,4): " << mainGrid[(4 * N * N) + (4 * N) + 4] << "°C" << std::endl;
    std::cout << "Front face center (4,4,0): " << mainGrid[(0 * N * N) + (4 * N) + 4] << "°C" << std::endl;
    std::cout << "Back face center (4,4,8): " << mainGrid[(8 * N * N) + (4 * N) + 4] << "°C" << std::endl;
    std::cout << "Side edge (0,4,4): " << mainGrid[(4 * N * N) + (4 * N) + 0] << "°C" << std::endl;

    // Output results in CSV format
    std::cout << "\nFinal temperature distribution:" << std::endl;
    std::cout << "x,y,z,temperature" << std::endl;
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                std::cout << x << "," << y << "," << z << "," 
                          << mainGrid[(z * N * N) + (y * N) + x] << std::endl;
            }
        }
    }

    // Cleanup
    cudaFree(deviceLocalGrids);
    cudaFree(deviceBlocks);
    cudaFree(deviceMainGrid);
    cudaFree(deviceConverged);

    delete[] hostLocalGrids;
    delete[] mainGrid;

    return 0;
}