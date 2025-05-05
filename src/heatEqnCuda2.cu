#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <string>

// Global variables for grid dimensions and cut coordinates
int ROWS, COLS, NUM_X_CUTS, NUM_Y_CUTS;
std::string FILE_NAME;
const double EPS = 1e-3;
std::vector<float> X_CORDS(0), Y_CORDS(0);

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
    float r_x = alpha * dt/(2 * dx * dx);
    float r_y = alpha * dt/(2 * dy * dy);
    float tempDiff = 100.0;
    float maxTempDiff = 100.0;
    std::vector<std::vector<float>> localGrid;
    
    // Constructor with built-in boundary trimming
    BlockOfGrid(int x_min, int x_max, int y_min, int y_max) 
        : xMin(x_min), 
          xMax(x_max), 
          yMin(y_min), 
          yMax(y_max) {

        localGrid.resize(xMax - xMin, std::vector<float>(yMax - yMin, 0));
    }

    __device__ void computeNextStateAll(const std::vector<std::vector<float>>& grid) {
        //Computes the next state for all the blocks including the boundary blocks. For boundary blocks, it trims the edges
        maxTempDiff = 0.0;
        //Use OMP here rather than main
        for (int i = std::max(1, xMin); i < std::min(xMax, ROWS - 1); ++i) {
            for (int j = std::max(1, yMin); j < std::min(yMax, COLS -1); ++j) {
                tempDiff = r_x * (grid[i+1][j] - 2*grid[i][j] + grid[i-1][j]) + r_y * (grid[i][j+1] - 2*grid[i][j] + grid[i][j-1]);
                localGrid[i - xMin][j - yMin] = grid[i][j] + tempDiff;
                maxTempDiff = std::max(maxTempDiff, tempDiff);
            }
        }
        if (xMin > 0 && xMax < ROWS && yMin > 0 && yMax < COLS) {
            return;
        }

        // Set edge cells to 0 only if the block touches a boundary
        if (xMin == 0) {  // Top boundary
            for (int j = yMin; j < yMax; ++j) {
                localGrid[0][j - yMin] = 100; // Accessing the top row of localGrid
            }
        }
        if (xMax == ROWS) {  // Bottom boundary
            for (int j = yMin; j < yMax; ++j) {
                localGrid[xMax - xMin - 1][j - yMin] = 100; // Accessing the bottom row of localGrid
            }
        }
        if (yMin == 0) {  // Left boundary
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][0] = 0; // Accessing the left column of localGrid
            }
        }
        if (yMax == COLS) {  // Right boundary
            for (int i = xMin; i < xMax; ++i) {
                localGrid[i - xMin][yMax - yMin - 1] = 0; // Accessing the right column of localGrid
            }
        }
    }
          
    // Function to update the global grid with the current block's state
    __device__ void updateGlobalGrid(std::vector<std::vector<float>>& grid) {
        //#pragma omp parallel for schedule(runtime)
        for (int i = xMin; i < xMax; ++i) {  // Ensures xMax - 1 is included
            for (int j = yMin; j < yMax; ++j) {  // Ensures yMax - 1 is included
                grid[i][j] = localGrid[i - xMin][j - yMin];
            }
        }
    }
};

// Function to initialize the grid and read input parameters
void initializeGrid(int argc, char* argv[]) {
    if (argc < 6) { // At least 4 values needed before coordinates
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <num_x_cords> <num_y_cords> <X1> ... <Xn> <Y1> ... <Ym>" << std::endl;
        exit(1);
    }

    //File name
    FILE_NAME = argv[1];

    // Read grid dimensions
    ROWS = std::atoi(argv[2]);
    COLS = std::atoi(argv[3]);

    // Read number of X and Y coordinates
    NUM_X_CUTS = std::atoi(argv[4]);
    NUM_Y_CUTS = std::atoi(argv[5]);

    int expectedArgs = 6 + NUM_X_CUTS + NUM_Y_CUTS;
    if (argc != expectedArgs) {
        std::cerr << "Error: Expected " << expectedArgs - 1 << " arguments, but received " << argc - 1 << "." << std::endl;
        exit(1);
    }

    // Read X coordinates
    X_CORDS.push_back(0);
    for (int i = 0; i < NUM_X_CUTS; i++) {
        X_CORDS.push_back(std::atoi(argv[i+6]));
    }
    X_CORDS.push_back(ROWS);

    // Read Y coordinates
    Y_CORDS.push_back(0);
    for (int i = 0; i < NUM_Y_CUTS; i++) {
        Y_CORDS.push_back(std::atoi(argv[i+6+NUM_X_CUTS]));
    }
    Y_CORDS.push_back(COLS);
} 

void saveCSVFile(const std::vector<std::vector<float>>& matrix, int timeStep, float delta,const std::string& filename){
    std::ofstream file(filename + "_" + std::to_string(timeStep) +".csv");
    file << "X,Y,Temperature" << "\n";

    for (int i = 0; i < ROWS; ++i){
        for (int j = 0; j < COLS; ++j){
            file << i << "," << j << "," << matrix[i][j] << "\n";
        }
    }

    file.close();
}

__global__ void simulateHeat(BlockOfGrid* block, )

int main(int argc, char* argv[]){
    initializeGrid(argc, argv);
    std::vector<std::vector<float>> mainGrid(ROWS, std::vector<float>(COLS, 0));
    double stopCriterion = 1.0;
    double localMax;
    int step = 0;

    // Create blocks of the grid based on the cut coordinates
    std::vector<BlockOfGrid> blocks;
    for (int i = 0; i <= NUM_X_CUTS; ++i) {
        for (int j = 0; j <= NUM_Y_CUTS; ++j) {
            blocks.emplace_back(X_CORDS[i], X_CORDS[i+1], Y_CORDS[j], Y_CORDS[j+1]);
        }
    }

    do{
        stopCriterion = 0.0;

        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i].computeNextStateAll(mainGrid);
            stopCriterion = blocks[i].maxTempDiff;
        }
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i].updateGlobalGrid(mainGrid);
        }
        ++step;
    }while (abs(stopCriterion) > EPS || step <= 5);
    saveCSVFile(mainGrid, step-1, stopCriterion, FILE_NAME);
    return 0;
}
