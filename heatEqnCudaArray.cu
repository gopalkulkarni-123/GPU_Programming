#include <iostream>
#include <cstdlib>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <vector>

// Global variables for grid dimensions and cut coordinates
int ROWS, COLS, NUM_X_CUTS, NUM_Y_CUTS;
std::string FILE_NAME;
const double EPS = 1e-3;
float X_CORDS[100], Y_CORDS[100];

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

    __device__ void computeNextStateAll(const float grid[100][100]) {
        maxTempDiff = 0.0;
        for (int i = std::max(1, xMin); i < std::min(xMax, ROWS - 1); ++i) {
            for (int j = std::max(1, yMin); j < std::min(yMax, COLS - 1); ++j) {
                tempDiff = r_x * (grid[i + 1][j] - 2 * grid[i][j] + grid[i - 1][j]) + r_y * (grid[i][j + 1] - 2 * grid[i][j] + grid[i][j - 1]);
                localGrid[i - xMin][j - yMin] = grid[i][j] + tempDiff;
                maxTempDiff = std::max(maxTempDiff, tempDiff);
            }
        }
        if (xMin > 0 && xMax < ROWS && yMin > 0 && yMax < COLS) {
            return;
        }

        if (xMin == 0) {
            for (int j = yMin; j < yMax; ++j) {
                localGrid[0][j - yMin] = 100;
            }
        }
        if (xMax == ROWS) {
            for (int j = yMin; j < yMax; ++j) {
                localGrid[xMax - xMin - 1][j - yMin] = 100;
            }
        }
        if (yMin == 0) {
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][0] = 0;
            }
        }
        if (yMax == COLS) {
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][yMax - yMin - 1] = 0;
            }
        }
    }

    __device__ void updateGlobalGrid(float grid[100][100]) {
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

void initializeGrid(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <num_x_cords> <num_y_cords> <X1> ... <Xn> <Y1> ... <Ym>" << std::endl;
        exit(1);
    }

    FILE_NAME = argv[1];

    ROWS = std::atoi(argv[2]);
    COLS = std::atoi(argv[3]);

    NUM_X_CUTS = std::atoi(argv[4]);
    NUM_Y_CUTS = std::atoi(argv[5]);

    int expectedArgs = 6 + NUM_X_CUTS + NUM_Y_CUTS;
    if (argc != expectedArgs) {
        std::cerr << "Error: Expected " << expectedArgs - 1 << " arguments, but received " << argc - 1 << "." << std::endl;
        exit(1);
    }

    X_CORDS[0] = 0;
    for (int i = 0; i < NUM_X_CUTS; i++) {
        X_CORDS[i + 1] = std::atoi(argv[i + 6]);
    }
    X_CORDS[NUM_X_CUTS + 1] = ROWS;

    Y_CORDS[0] = 0;
    for (int i = 0; i < NUM_Y_CUTS; i++) {
        Y_CORDS[i + 1] = std::atoi(argv[i + 6 + NUM_X_CUTS]);
    }
    Y_CORDS[NUM_Y_CUTS + 1] = COLS;
}

void saveCSVFile(const float matrix[100][100], int timeStep, float delta, const std::string& filename) {
    std::ofstream file(filename + "_" + std::to_string(timeStep) + ".csv");
    file << "X,Y,Temperature" << "\n";

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            file << i << "," << j << "," << matrix[i][j] << "\n";
        }
    }

    file.close();
}

__global__ void simulateHeat(BlockOfGrid* individualBlock, float grid[100][100], int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < numBlocks; i += blockDim.x * gridDim.x) {
        individualBlock[i].computeNextStateAll(grid);  // Pass the grid here
    }
}

int main(int argc, char* argv[]) {
    initializeGrid(argc, argv);
    float mainGrid[100][100] = {0};
    double stopCriterion = 1.0;
    double localMax;
    int step = 0;
    int blockSize = 256;  // Threads per block
    int numBlocks = blocks.size();  // Use blocks.size() for numBlocks
    BlockOfGrid* d_individualBlock;

    std::vector<BlockOfGrid> blocks;
    for (int i = 0; i <= NUM_X_CUTS; ++i) {
        for (int j = 0; j <= NUM_Y_CUTS; ++j) {
            blocks.emplace_back(X_CORDS[i], X_CORDS[i + 1], Y_CORDS[j], Y_CORDS[j + 1]);
        }
    }

    CUDA_CHECK(cudaMalloc((void**)&d_individualBlock, sizeof(BlockOfGrid) * blocks.size()));

    CUDA_CHECK(cudaMemcpy(d_individualBlock, blocks.data(), sizeof(BlockOfGrid) * blocks.size(), cudaMemcpyHostToDevice));

    // Launch the kernel
    simulateHeat<<<numBlocks, blockSize>>>(d_individualBlock, mainGrid, blocks.size());
    CUDA_CHECK(cudaMemcpy(blocks.data(), d_individualBlock, sizeof(BlockOfGrid) * blocks.size(), cudaMemcpyDeviceToHost));

    // Copy the result back to the host
    for (int i = 0; i < blocks.size(); ++i) {
        blocks[i].updateGlobalGrid(mainGrid);
    }

    CUDA_CHECK(cudaFree(d_individualBlock));
    saveCSVFile(mainGrid, step - 1, stopCriterion, FILE_NAME);

    std::cout << "Simulation completed. Results saved to " << FILE_NAME << "_" << step - 1 << ".csv" << std::endl;
    return 0;
}
