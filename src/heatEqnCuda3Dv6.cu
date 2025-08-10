#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <vector>

#define Nx 100
#define Ny 100
#define Nz 100

#define MAX_ITER 10000

#define numCutsX 2
#define numCutsY 2
#define numCutsZ 1


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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cudaSetDevice(world_rank);  // One GPU per rank

    // Full grid allocation
    size_t fullGridSize = Nx * Ny * Nz;
    float* h_globalGrid;
    cudaHostAlloc(&h_globalGrid, fullGridSize * sizeof(float), cudaHostAllocDefault);

    // For device-local partial view
    std::vector<BlockOfGrid> ownedBlocks;

    const int xBlockSize = (Nx + numCutsX - 1) / numCutsX;
    const int yBlockSize = (Ny + numCutsY - 1) / numCutsY;
    const int zBlockSize = (Nz + numCutsZ - 1) / numCutsZ;

    int totalBlocks = numCutsX * numCutsY * numCutsZ;

    // Build all blocks; assign blocks to GPUs contiguously or round-robin
    for (int bz = 0; bz < numCutsZ; ++bz) {
        for (int by = 0; by < numCutsY; ++by) {
            for (int bx = 0; bx < numCutsX; ++bx) {
                int blockID = bz * numCutsY * numCutsX + by * numCutsX + bx;
                if (blockID % world_size != world_rank) continue;

                int xMin = bx * xBlockSize;
                int xMax = std::min((bx + 1) * xBlockSize, Nx);
                int yMin = by * yBlockSize;
                int yMax = std::min((by + 1) * yBlockSize, Ny);
                int zMin = bz * zBlockSize;
                int zMax = std::min((bz + 1) * zBlockSize, Nz);

                int lx = xMax - xMin;
                int ly = yMax - yMin;
                int lz = zMax - zMin;

                float* d_subgrid;
                cudaMalloc(&d_subgrid, lx * ly * lz * sizeof(float));
                cudaMemset(d_subgrid, 0, lx * ly * lz * sizeof(float));

                ownedBlocks.emplace_back(xMin, xMax, yMin, yMax, zMin, zMax, d_subgrid);
            }
        }
    }

    int localNumBlocks = ownedBlocks.size();

    // Copy blocks to device
    BlockOfGrid* d_blocks;
    cudaMalloc(&d_blocks, localNumBlocks * sizeof(BlockOfGrid));
    cudaMemcpy(d_blocks, ownedBlocks.data(), localNumBlocks * sizeof(BlockOfGrid), cudaMemcpyHostToDevice);

    // Local grid slice memory
    float* d_localGrid;
    cudaMalloc(&d_localGrid, fullGridSize * sizeof(float));
    cudaMemset(d_localGrid, 0, fullGridSize * sizeof(float));

    float* h_localGrid;
    cudaHostAlloc(&h_localGrid, fullGridSize * sizeof(float), cudaHostAllocDefault);

    // Shared grid setup for Allgatherv
    int* recvCounts = new int[world_size];
    int* displs = new int[world_size];

    int baseBlocks = totalBlocks / world_size;
    int extra = totalBlocks % world_size;

    for (int i = 0; i < world_size; ++i) {
        int numBlocks = baseBlocks + (i < extra ? 1 : 0);
        size_t localSize = 0;
        for (int b = i; b < totalBlocks; b += world_size) {
            int bz = b / (numCutsY * numCutsX);
            int by = (b / numCutsX) % numCutsY;
            int bx = b % numCutsX;

            int xMin = bx * xBlockSize;
            int xMax = std::min((bx + 1) * xBlockSize, Nx);
            int yMin = by * yBlockSize;
            int yMax = std::min((by + 1) * yBlockSize, Ny);
            int zMin = bz * zBlockSize;
            int zMax = std::min((bz + 1) * zBlockSize, Nz);

            localSize += (xMax - xMin) * (yMax - yMin) * (zMax - zMin);
        }
        recvCounts[i] = localSize;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvCounts[i - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        for (int i = 0; i < localNumBlocks; ++i) {
            processBlock<<<1, 128>>>(d_blocks, i, d_localGrid, localNumBlocks);
            cudaDeviceSynchronize();
        }

        // Copy updated localGrid to host (full size)
        cudaMemcpy(h_localGrid + displs[world_rank], d_localGrid + displs[world_rank],
                   recvCounts[world_rank] * sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Allgatherv(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,  // IN_PLACE since we already wrote into h_localGrid
            h_localGrid, recvCounts, displs, MPI_FLOAT,
            MPI_COMM_WORLD
        );

        // Copy the new global state back to device
        cudaMemcpy(d_localGrid, h_localGrid, fullGridSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Output final result from rank 0
    if (world_rank == 0) {
        std::ofstream csvFile("./output/temperature_output_multigpu.csv");
        if (!csvFile) {
            std::cerr << "Error: could not open file for writing.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        csvFile << "x,y,z,temperature\n";
        for (int z = 0; z < Nz; ++z)
            for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                    csvFile << x << "," << y << "," << z << ","
                            << h_localGrid[z * Nx * Ny + y * Nx + x] << "\n";

        std::cout << "Execution time: " << float(duration.count()) / 1e6 << " seconds\n";
    }

    // Cleanup
    for (auto& blk : ownedBlocks)
        cudaFree(blk.localGrid);

    cudaFree(d_blocks);
    cudaFree(d_localGrid);
    cudaFreeHost(h_localGrid);
    cudaFreeHost(h_globalGrid);
    delete[] recvCounts;
    delete[] displs;

    MPI_Finalize();
    return 0;
}
