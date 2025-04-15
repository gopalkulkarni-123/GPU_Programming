#include <iostream>
#include <cstdlib>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <vector>

// Global variables for grid dimensions and cut coordinates
std::string FILE_NAME;
const double EPS = 1e-3;

// Structure to represent a block of the grid
struct BlockOfGrid {
    int xMin;
    int xMax;
    int yMin;
    int yMax;
    float alpha = 0.75;
    float dx = 1.0;
    float dy = 1.0;
    float dt = 0.1;
    float r_x = alpha * dt / (2 * dx * dx);
    float r_y = alpha * dt / (2 * dy * dy);
    float tempDiff = 100.0;
    float maxTempDiff = 100.0;
    int d_rows = 100
    int d_cols = 100
    float localGrid[100][100];

    // Constructor with built-in boundary trimming
    BlockOfGrid(int x_min, int x_max, int y_min, int y_max)
        : xMin(x_min),
          xMax(x_max),
          yMin(y_min),
          yMax(y_max) {
        for (int i = 0; i < xMax - xMin; ++i) {
            for (int j = 0; j < yMax - yMin; ++j) {
                localGrid[i][j] = 0;
            }
        }
    }

    __device__ inline int d_max(int a, int b) {
        return a > b ? a : b;
}

    __device__ inline int d_min(int a, int b) {
        return a < b ? a : b;
    }

    __device__ void computeNextStateAll(const float grid[100][100]) {
        maxTempDiff = 0.0;
        for (int i = d_max(1, xMin); i < d_min(xMax, d_rows - 1); ++i) {
            for (int j = d_max(1, yMin); j < d_min(yMax, d_cols - 1); ++j) {
                tempDiff = r_x * (grid[i + 1][j] - 2 * grid[i][j] + grid[i - 1][j]) + r_y * (grid[i][j + 1] - 2 * grid[i][j] + grid[i][j - 1]);
                localGrid[i - xMin][j - yMin] = grid[i][j] + tempDiff;
                maxTempDiff = d_max(maxTempDiff, tempDiff);
            }
        }
        if (xMin > 0 && xMax < 100 && yMin > 0 && yMax < 100) {
            return;
        }

        if (xMin == 0) {
            for (int j = yMin; j < yMax; ++j) {
                localGrid[0][j - yMin] = 100;
            }
        }
        if (xMax == 100) {
            for (int j = yMin; j < yMax; ++j) {
                localGrid[xMax - xMin - 1][j - yMin] = 100;
            }
        }
        if (yMin == 0) {
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][0] = 0;
            }
        }
        if (yMax == 100) {
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][yMax - yMin - 1] = 0;
            }
        }
    }

    void updateGlobalGrid(float grid[100][100]) {
        for (int i = xMin; i < xMax; ++i) {
            for (int j = yMin; j < yMax; ++j) {
                grid[i][j] = localGrid[i - xMin][j - yMin];
            }
        }
    }
};

inline void checkCudaError(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(result) checkCudaError(result, __FILE__, __LINE__)

void initializeGrid(float grid[100][100]){
    // Initialize the grid with some values
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            grid[i][j] = 0.0f;
        }
    }
    // Set boundary conditions
    for (int i = 0; i < 100; ++i) {
        grid[i][0] = 100.0f;  // Left boundary
        grid[i][100 - 1] = 100.0f;  // Right boundary
    }
    for (int j = 0; j < 100; ++j) {
        grid[0][j] = 100.0f;  // Top boundary
        grid[100 - 1][j] = 100.0f;  // Bottom boundary
    }
}

__global__ void simulateHeat(BlockOfGrid* individualBlock, float grid[100][100]){
    int idx = threadIdx.x;
    for (int i = idx; i < 4; i += blockDim.x * gridDim.x) {
        individualBlock[i].computeNextStateAll(grid);  // Pass the grid here
    }
}

int main() {
    float hostMainGrid [100][100];
    float (*deviceMainGrid)[100];
    initializeGrid(hostMainGrid);

    BlockOfGrid vectorOfBlocks[4] = {
        {0, 50, 0, 50},
        {0, 50, 50, 100},
        {50, 100, 0, 50},
        {50, 100, 50, 100}
    };

    //Allocatre device memory
    BlockOfGrid* deviceBlocks;
    cudaMalloc(&deviceBlocks, sizeof(BlockOfGrid) * 4);
    cudaMalloc(&deviceMainGrid, sizeof(float) * 100 * 100);

    //Copy data to device
    cudaMemcpy(deviceBlocks, vectorOfBlocks, sizeof(BlockOfGrid) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMainGrid, hostMainGrid, sizeof(float) * 100 * 100, cudaMemcpyHostToDevice);

    //Kernel Launch
    for (int i = 0; i < 1000; ++i){
        simulateHeat<<<1, 4>>>(deviceBlocks, deviceMainGrid);
        cudaMemcpy(hostMainGrid, deviceMainGrid, sizeof(float) * 100 * 100, cudaMemcpyDeviceToHost);
    }
    std::cout << "Simulation completed";
    cudaFree(deviceBlocks);
    cudaFree(deviceMainGrid);
    return 0;
}