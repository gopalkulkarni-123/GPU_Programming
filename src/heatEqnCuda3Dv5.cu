#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>

#define Nx 100
#define Ny 10
#define Nz 10

#define MAX_ITER 10000

//#define numCutsX 20
//#define numCutsY 20
//#define numCutsZ 20


struct BlockOfGrid {
    int xMin, xMax, yMin, yMax, zMin, zMax;
    int blockLength, blockBreadth, blockDepth;
    float* localGrid;

    //Boundary flags
    bool isOnXMinBoundary, isOnXMaxBoundary;
    bool isOnYMinBoundary, isOnYMaxBoundary;
    bool isOnZMinBoundary, isOnZMaxBoundary;
    bool isInteriorBlock;

    // Physical constants
    float alpha = 0.35f;
    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    float dt = 0.1f;
    float r_x = (alpha * dt) / (dx * dx);
    float r_y = (alpha * dt) / (dy * dy);
    float r_z = (alpha * dt) / (dz * dz);

    __host__ __device__ BlockOfGrid(int x_min = 0, int x_max = 0, int y_min = 0, int y_max = 0, int z_min = 0, int z_max = 0, float* gridPtr = nullptr)
        : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max), zMin(z_min), zMax(z_max), localGrid(gridPtr), 
        blockLength(y_max - y_min), blockBreadth(x_max - x_min), blockDepth(z_max - z_min) {

        // Determine which boundaries this block touches
        isOnXMinBoundary = (xMin == 0);
        isOnXMaxBoundary = (xMax == Nx);
        isOnYMinBoundary = (yMin == 0);
        isOnYMaxBoundary = (yMax == Ny);
        isOnZMinBoundary = (zMin == 0);
        isOnZMaxBoundary = (zMax == Nz);

        // Check if this is a completely interior block
        isInteriorBlock = !isOnXMinBoundary && !isOnXMaxBoundary && 
                         !isOnYMinBoundary && !isOnYMaxBoundary && 
                         !isOnZMinBoundary && !isOnZMaxBoundary;
    }
    
    __device__ void compute(float* dGrid){
        // First, copy data to local grid using grid-stride loop
        int totalPoints = blockLength * blockBreadth * blockDepth;
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < totalPoints;
         idx += blockDim.x * gridDim.x) {

            // Convert 1D idx to 3D (x, y, z) within the block
            /* x = idx % blockLength;
            int y = (idx / blockLength) % blockBreadth;
            int z = idx / (blockLength * blockBreadth);*/

            int x = idx % blockBreadth;
            int y = (idx / blockBreadth) % blockLength;
            int z = idx / (blockBreadth * blockLength);

            // Compute global indices
            int globalX = xMin + x;
            int globalY = yMin + y;
            int globalZ = zMin + z;

            // Flattened global index: (z * Y * X) + (y * X) + x
            int globalIdx = (globalZ * Ny * Nx) + (globalY * Nx) + globalX;

            // Flattened local index: (z * blockBreadth * blockLength) + (y * blockLength) + x
            int localIdx = (z * blockLength * blockBreadth) + (y * blockBreadth) + x;

            //printf("%d, %d, %d, %d, %d \n", x, y, z, localIdx, globalIdx);

            localGrid[localIdx] = dGrid[globalIdx];
        }
        
        // Synchronize threads to ensure all data is copied
        __syncthreads();
        
        // Apply heat equation - optimized based on block type
        if (isInteriorBlock) {
            computeInteriorOnly(dGrid);
        } else {
            computeWithBoundaryChecks(dGrid);
        }
        
        // Synchronize threads after computation
        __syncthreads();
        
        // Apply boundary conditions only if this block touches boundaries
        if (!isInteriorBlock) {
            applyBoundaryConditions();
        }
        
        // Synchronize threads after boundary conditions
        __syncthreads();
        
        // Copy results back to global grid using grid-stride loop
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; 
             idx < totalPoints; 
             idx += blockDim.x * gridDim.x) {
            
            /*int x = idx % blockLength;
            int y = (idx / blockLength) % blockBreadth;
            int z = idx / (blockLength * blockBreadth);*/

            int x = idx % blockBreadth;
            int y = (idx / blockBreadth) % blockLength;
            int z = idx / (blockBreadth * blockLength);

            int globalX = xMin + x;
            int globalY = yMin + y;
            int globalZ = zMin + z;

            int globalIdx = (globalZ * Ny * Nx) + (globalY * Nx) + globalX;

            int localIdx = (z * blockBreadth * blockLength) + (y * blockBreadth) + x;
            
            dGrid[globalIdx] = localGrid[localIdx];
        }
    }

    __device__ void computeInteriorOnly(float* dGrid) {
        int totalPoints = blockLength * blockBreadth * blockDepth;
        
        // Grid-stride loop for interior computation
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; 
             idx < totalPoints; 
             idx += blockDim.x * gridDim.x) {
            
            // Convert 1D idx to 2D (x, y) within the block
            /*int x = idx % blockLength;
            int y = (idx / blockLength) % blockBreadth;
            int z = (idx /(blockLength * blockBreadth));*/

            int x = idx % blockBreadth;
            int y = (idx / blockBreadth) % blockLength;
            int z = idx / (blockBreadth * blockLength);
            
            // Global coordinates
            int globalX = xMin + x;
            int globalY = yMin + y;
            int globalZ = zMin + z;
            
            // Skip points on global boundaries (they don't get updated)
            /*if (globalX <= 0 || globalX >= Nx-1 || 
                globalY <= 0 || globalY >= Ny-1 ||
                globalZ <= 0 || globalZ >= Nz-1) {
                continue;
            }*/
            
            // Apply finite difference heat equation using global grid values
            float center = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + globalX];
            float left   = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + (globalX - 1)];
            float right  = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + (globalX + 1)];
            float top    = dGrid[(globalZ * Ny * Nx) + ((globalY + 1) * Nx) + globalX];
            float bottom = dGrid[(globalZ * Ny * Nx) + ((globalY - 1) * Nx) + globalX];
            float front  = dGrid[((globalZ + 1) * Ny * Nx) + (globalY * Nx) + globalX];
            float back   = dGrid[((globalZ - 1) * Ny * Nx) + (globalY * Nx) + globalX];

            
            float tempDiff = r_x * (left - 2.0f * center + right) +
                            r_y * (bottom - 2.0f * center + top) +
                            r_z * (back - 2.0f * center + front);
            
            float newValue = center + tempDiff;
            int localIdx = (z * blockLength * blockBreadth) + (y * blockBreadth) + x;
            localGrid[localIdx] = newValue;
        }
    }

    __device__ void computeWithBoundaryChecks(float* dGrid) {
        int totalPoints = blockLength * blockBreadth * blockDepth;
        
        // Grid-stride loop for interior computation
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; 
             idx < totalPoints; 
             idx += blockDim.x * gridDim.x) {
            
            // Convert 1D idx to 2D (x, y) within the block
            /*int x = idx % blockLength;
            int y = (idx / blockLength) % blockBreadth;
            int z = (idx /(blockLength * blockBreadth));*/

            int x = idx % blockBreadth;
            int y = (idx / blockBreadth) % blockLength;
            int z = idx / (blockBreadth * blockLength);
            
            // Global coordinates
            int globalX = xMin + x;
            int globalY = yMin + y;
            int globalZ = zMin + z;
            
            // Skip points on global boundaries (they don't get updated)
            if (globalX <= 0 || globalX >= Nx-1 || 
                globalY <= 0 || globalY >= Ny-1 ||
                globalZ <= 0 || globalZ >= Nz-1) {
                continue;
            }
            
            // Apply finite difference heat equation using global grid values
            float center = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + globalX];
            float left   = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + (globalX - 1)];
            float right  = dGrid[(globalZ * Ny * Nx) + (globalY * Nx) + (globalX + 1)];
            float top    = dGrid[(globalZ * Ny * Nx) + ((globalY + 1) * Nx) + globalX];
            float bottom = dGrid[(globalZ * Ny * Nx) + ((globalY - 1) * Nx) + globalX];
            float front  = dGrid[((globalZ + 1) * Ny * Nx) + (globalY * Nx) + globalX];
            float back   = dGrid[((globalZ - 1) * Ny * Nx) + (globalY * Nx) + globalX];
            
            float tempDiff = r_x * (left - 2.0f * center + right) +
                            r_y * (bottom - 2.0f * center + top) +
                            r_z * (back - 2.0f * center + front);
            
            float newValue = center + tempDiff;
            int localIdx = (z * blockLength * blockBreadth) + (y * blockBreadth) + x;
            localGrid[localIdx] = newValue;
            }
        }



    __device__ void applyBoundaryConditions() {
        int totalPoints = blockLength * blockBreadth * blockDepth;
        
        // Grid-stride loop for boundary conditions
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; 
             idx < totalPoints; 
             idx += blockDim.x * gridDim.x) {
            
            // Convert 1D idx to 2D (x, y) within the block
            /*int x = idx % blockLength;
            int y = (idx / blockLength) % blockBreadth;
            int z = (idx /(blockLength * blockBreadth));*/

            int x = idx % blockBreadth;
            int y = (idx / blockBreadth) % blockLength;
            int z = idx / (blockBreadth * blockLength);
            
            // Global coordinates
            int globalX = xMin + x;
            int globalY = yMin + y;
            int globalZ = zMin + z;
            
            int localIdx = (z * blockLength * blockBreadth) + (y * blockBreadth) + x;
            
            // Check if this point is on a global boundary
            bool onBoundary = false;
            float boundaryValue = 0.0f;
            
            // Fixed temperature boundaries
            if (globalX == 0 || globalX == Nx-1) {
                onBoundary = true;
                boundaryValue = 0.0f;  // Cold boundaries
            }
            else if (globalY == 0 || globalY == Ny-1) {
                onBoundary = true;
                boundaryValue = 100.0f;  // Hot boundaries (heat sources)
            }
            else if (globalZ == 0 || globalZ == Nz-1){
                onBoundary = true;
                boundaryValue = 0.0f;
            }
            
            if (onBoundary) {
                localGrid[localIdx] = boundaryValue;
            }
        }
    }
};

__global__ void processBlock(BlockOfGrid* blocks, int blockIndex, float* Grid, int totalBlocks) {
    int idx = blockIdx.x;
    if(idx < totalBlocks){
        blocks[blockIndex].compute(Grid);
    }
}

int main() {
    // Initialize grid
    float* h_grid = new float[Nx*Ny*Nz];
    float* d_grid;
    cudaMalloc(&d_grid, Nx*Ny*Nz*sizeof(float));
    
    // Initialize with some values
    for (int i = 0; i < Nx*Ny*Nz; i++) {
        h_grid[i] = 0.0f;
    }
    cudaMemcpy(d_grid, h_grid, Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);

    const int cuts[] = {1, 2, 4, 5, 10, 20};

    for (int cutsIter = 0; cutsIter < 1; ++cutsIter){
        float totalTime = 0.0f;
        float avgTime = 0.0f;
        for (int repeatIter = 0; repeatIter < 1; ++repeatIter){
            int numCutsX = cuts[cutsIter];
            int numCutsY = cuts[cutsIter];
            //int numCutsZ = cuts[cutsIter];
            int numCutsZ = 10;

            // Create blocks (10x10 blocks covering the 100x100 grid)
            const int xBlockSize = (Nx + numCutsX - 1) / numCutsX;
            const int yBlockSize = (Ny + numCutsY - 1) / numCutsY;
            const int zBlockSize = (Nz + numCutsZ - 1) / numCutsZ;
            const int totalBlocks = numCutsX * numCutsY * numCutsZ;
            
            BlockOfGrid* h_blocks = new BlockOfGrid[totalBlocks];
            BlockOfGrid* d_blocks;
            cudaMalloc(&d_blocks, totalBlocks*sizeof(BlockOfGrid));
            
            // Initialize each block
            for(int bz = 0; bz < numCutsZ; bz++){
                for (int by = 0; by < numCutsY; by++) {
                    for (int bx = 0; bx < numCutsX; bx++) {
                        int xMin = bx * xBlockSize;
                        int xMax = min((bx + 1) * xBlockSize, Nx);
                        int yMin = by * yBlockSize;
                        int yMax = min((by + 1) * yBlockSize, Ny);
                        int zMin = bz * zBlockSize;
                        int zMax = min((bz + 1) * zBlockSize, Nz);

                        int blockLength = yMax - yMin;
                        int blockBreadth = xMax - xMin;
                        int blockDepth = zMax - zMin;
                        
                        // Allocate local grid for each block
                        float* d_localGrid;
                        cudaMalloc(&d_localGrid, blockLength * blockBreadth * blockDepth * sizeof(float));
                        
                        h_blocks[((bz * numCutsY * numCutsX)+ (by * numCutsX) + bx)] = BlockOfGrid(
                            xMin, xMax, yMin, yMax, zMin, zMax, d_localGrid
                        );
                    }
                }
            }
            
            // Copy blocks to device
            cudaMemcpy(d_blocks, h_blocks, totalBlocks*sizeof(BlockOfGrid), cudaMemcpyHostToDevice);

            // Process blocks - use more threads to make grid-stride loops effective
            //dim3 gridDim(1, 1);
            //dim3 blockDim(128, 1);  // Use more threads for grid-stride loops
            
            /*std::cout << "Before Computation \n";
            for(int i = 0; i < totalBlocks; ++i){
                std::cout <<"Block id = "<< i << "\n" 
                << "x_min = " << h_blocks[i].xMin << "\n"
                << "x_max = " << h_blocks[i].xMax << "\n"
                << "y_min = " << h_blocks[i].yMin << "\n"
                << "y_max = " << h_blocks[i].yMax << "\n"
                << "z_min = " << h_blocks[i].zMin << "\n"
                << "z_max = " << h_blocks[i].zMax << "\n"
                <<"---------------------------------- \n";                    
            }*/
            
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < MAX_ITER; iter++) {
                for (int blockNum = 0; blockNum < totalBlocks; ++blockNum){
                    processBlock<<<totalBlocks, 128>>>(d_blocks, blockNum, d_grid, totalBlocks);
                    cudaDeviceSynchronize();
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            cudaMemcpy(h_grid, d_grid, Nx*Ny*Nz*sizeof(float), cudaMemcpyDeviceToHost);
            
            std::ostringstream oss;
            oss << "./output/temperature_output_pipe_" << numCutsX << ".csv";

            std::ofstream csvFile(oss.str());

            if (!csvFile) {
                std::cerr << "Error: could not open file for writing.\n";
                return 1;
            }

            csvFile << "x,y,z,temperature\n";

            for (int z = 0; z < Nz; ++z) {
                for (int x = 0; x < Nx; ++x) {
                    for (int y = 0; y < Ny; ++y) {
                        csvFile << x << "," << y << "," << z << ","
                                << h_grid[(z * Ny * Nx) + (y * Nx) + x] << "\n";
                    }
                }
            }

            std::cout << "Cuts:" << numCutsX << "; Iteration:" << repeatIter <<"; Execution time: " << float(duration.count() / 1000000.0) << " seconds\n";
            totalTime += float(duration.count() / 1000000.0);

            for (int i = 0; i < totalBlocks; i++) {
                cudaFree(h_blocks[i].localGrid);
            }
            delete[] h_blocks;
            cudaFree(d_blocks);
        }
        avgTime = totalTime/ 10;
        std::cout << "Average time for 10 iterations for cuts:" << cutsIter << " = " << avgTime << " seconds" << std::endl;
        std::cout << "---------------------------------------------\n";
    }
    
    // Cleanup
    cudaFree(d_grid);
    delete[] h_grid;
    
    return 0;
}