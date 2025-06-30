#include <iostream>
#include <cuda_runtime.h>

// Define matrix dimensions (assumed to be multiples of 32 for simplicity)
const int ROWS = 1024;
const int COLS = 1024;
const int THREADS_PER_BLOCK = 256;

// Warp size (32 threads)
const int WARP_SIZE = 32;

// Bitwise operation types
enum BitwiseOp { AND, OR, XOR };

// Kernel: Perform bitwise operation between two matrices using warp-level primitives
__global__ void bitwiseMatrixOp(const uint32_t* A, const uint32_t* B, uint32_t* C, BitwiseOp op) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ROWS && col < COLS) {
        // Load A and B values (coalesced access)
        uint32_t a_val = A[row * COLS + col];
        uint32_t b_val = B[row * COLS + col];
        uint32_t c_val;

        // Perform the requested bitwise operation
        switch (op) {
            case AND: c_val = a_val & b_val; break;
            case OR:  c_val = a_val | b_val; break;
            case XOR: c_val = a_val ^ b_val; break;
        }

        // Optional: Warp-level reduction (e.g., XOR across warp)
        // This is useful if you need to aggregate results across threads
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            c_val ^= __shfl_down_sync(0xFFFFFFFF, c_val, offset);
        }

        // Store result (coalesced write)
        C[row * COLS + col] = c_val;
    }
}

// Helper function to initialize a matrix with random bits
void initMatrix(uint32_t* matrix) {
    for (int i = 0; i < ROWS * COLS; i++) {
        matrix[i] = rand() & 0xFFFFFFFF; // Random 32-bit values
    }
}

int main() {
    // Allocate host matrices
    uint32_t *h_A = new uint32_t[ROWS * COLS];
    uint32_t *h_B = new uint32_t[ROWS * COLS];
    uint32_t *h_C = new uint32_t[ROWS * COLS];

    // Initialize with random bits
    initMatrix(h_A);
    initMatrix(h_B);

    // Allocate device matrices
    uint32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, ROWS * COLS * sizeof(uint32_t));
    cudaMalloc(&d_B, ROWS * COLS * sizeof(uint32_t));
    cudaMalloc(&d_C, ROWS * COLS * sizeof(uint32_t));

    // Copy data to device
    cudaMemcpy(d_A, h_A, ROWS * COLS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ROWS * COLS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(32, 8); // 256 threads per block (32x8)
    dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);

    // Launch kernel (perform XOR between A and B)
    bitwiseMatrixOp<<<grid, block>>>(d_A, d_B, d_C, XOR);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, ROWS * COLS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Verify correctness (check first 10 elements)
    for (int i = 0; i < 10; i++) {
        printf("A[%d] = %08X, B[%d] = %08X, C[%d] = %08X\n",
               i, h_A[i], i, h_B[i], i, h_C[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}