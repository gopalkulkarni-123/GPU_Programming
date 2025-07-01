#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

// Grid parameters
const int Nx = 150;
const int Ny = 150;
const int Nz = 150;

const double dx = 1.0f;
const double dy = 1.0f;
const double dz = 1.0f;

const double alpha = 0.35f;
const double dt = 0.1f; // CFL condition
const int steps = 1000;

// Convert 3D index to 1D index
inline int idx(int x, int y, int z) {
    return x + Nx * (y + Ny * z);
}

// Apply Dirichlet boundary conditions
void applyBoundaryConditions(std::vector<double>& T) {
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            T[idx(0, y, z)] = 0.0;         // x = 0 face
            T[idx(Nx - 1, y, z)] = 0.0;     // x = Nx - 1 face
        }
        for (int x = 0; x < Nx; ++x) {
            T[idx(x, 0, z)] = 100.0;           // y = 0 face
            T[idx(x, Ny - 1, z)] = 100.0;       // y = Ny - 1 face
        }
    }

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            T[idx(x, y, 0)] = 0.0;           // z = 0 face
            T[idx(x, y, Nz - 1)] = 0.0;       // z = Nz - 1 face
        }
    }
}

// Print final temperature as CSV to stdout
void printTemperatureCSV(const std::vector<double>& T) {
    std::cout << "x,y,z,temperature\n";
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                std::cout << x << "," << y << "," << z << "," << T[idx(x, y, z)] << "\n";
            }
        }
    }
}

int main() {
    using namespace std::chrono;

    auto start_time = high_resolution_clock::now();

    // Initialize temperature field to 20°C
    std::vector<double> T(Nx * Ny * Nz, 20.0);
    std::vector<double> T_new(Nx * Ny * Nz, 20.0);

    // Apply initial boundary conditions
    applyBoundaryConditions(T);

    // Time stepping loop
    for (int step = 0; step < steps; ++step) {
        for (int z = 1; z < Nz - 1; ++z) {
            for (int y = 1; y < Ny - 1; ++y) {
                for (int x = 1; x < Nx - 1; ++x) {
                    int i = idx(x, y, z);

                    double d2Tdx2 = (T[idx(x + 1, y, z)] - 2 * T[i] + T[idx(x - 1, y, z)]) / (dx * dx);
                    double d2Tdy2 = (T[idx(x, y + 1, z)] - 2 * T[i] + T[idx(x, y - 1, z)]) / (dy * dy);
                    double d2Tdz2 = (T[idx(x, y, z + 1)] - 2 * T[i] + T[idx(x, y, z - 1)]) / (dz * dz);

                    T_new[i] = T[i] + alpha * dt * (d2Tdx2 + d2Tdy2 + d2Tdz2);
                }
            }
        }

        std::swap(T, T_new);
        applyBoundaryConditions(T);
    }

    auto end_time = high_resolution_clock::now();
    duration<double> elapsed = end_time - start_time;

    // Output final temperature field
    printTemperatureCSV(T);

    // Print timing to stderr
    std::cerr << "Simulation time: " << elapsed.count() << " seconds\n";

    return 0;
}

