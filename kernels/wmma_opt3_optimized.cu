#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA configuration
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILE_M 128
#define TILE_N 128
#define TILE_K WMMA_K
#define STAGES 2  // Double buffering
#define FRAGS_PER_THREAD 4  // Critical for latency hiding

// Shared memory padding to avoid bank conflicts
#define SMEM_PAD 8

__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Threadblock-level data
    const int warpId = threadIdx.x / 32;
    // const int laneId = threadIdx.x % 32;
    
    // Each warp computes a 64x16 portion of the tile
    const int warpRow = (warpId / 8) * WMMA_M * 2;  // 2 WMMA tiles per warp in M
    const int warpCol = (warpId % 8) * WMMA_N;      // 8 warps in N direction
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ half shmemA[TILE_M * TILE_K + SMEM_PAD];
    __shared__ half shmemB[TILE_K * TILE_N + SMEM_PAD];
    
    // Fragments for WMMA operations (simplified to single fragment per type)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Block position
    const int blockM = blockIdx.x * TILE_M;
    const int blockN = blockIdx.y * TILE_N;
    
    // Early exit if outside matrix bounds
    if (blockM >= M || blockN >= N) return;
    
    // Pointers to global memory
    // const half* globalA = A + blockM * K;
    // const half* globalB = B + blockN;
    
    // Main computation loop - simplified approach
    for (int k = 0; k < K; k += TILE_K) {
        // Cooperative loading of A and B tiles
        const int tid = threadIdx.x;
        const int threads_per_block = blockDim.x;
        
        // Load A tile cooperatively
        for (int i = tid; i < TILE_M * TILE_K; i += threads_per_block) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int globalRow = blockM + row;
            int globalCol = k + col;
            
            if (globalRow < M && globalCol < K) {
                shmemA[i] = A[globalRow * K + globalCol];
            } else {
                shmemA[i] = __float2half(0.0f);
            }
        }
        
        // Load B tile cooperatively  
        for (int i = tid; i < TILE_K * TILE_N; i += threads_per_block) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int globalRow = k + row;
            int globalCol = blockN + col;
            
            if (globalRow < K && globalCol < N) {
                shmemB[i] = B[globalRow * N + globalCol];
            } else {
                shmemB[i] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // WMMA computation
        if (warpRow + WMMA_M <= TILE_M && warpCol + WMMA_N <= TILE_N) {
            wmma::load_matrix_sync(a_frag, &shmemA[warpRow * TILE_K], TILE_K);
            wmma::load_matrix_sync(b_frag, &shmemB[warpCol], TILE_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    int cRow = blockM + warpRow;
    int cCol = blockN + warpCol;
    
    if (cRow < M && cCol < N && cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

extern "C" {
    void launch_wmma_opt3(half* d_A, half* d_B, float* d_C, int M, int N, int K, cudaStream_t stream = 0) {
        dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
        dim3 block(256);  // 8 warps (256 threads)
        
        wmma_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    }
}