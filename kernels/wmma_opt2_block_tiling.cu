#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Optimization Level 2: Block Tiling
__global__ void wmma_opt2_block_tiling(
    half* a, half* b, float* c,
    int m, int n, int k
) {
    const int BLOCK_SIZE = 64;
    
    __shared__ half shmem_a[BLOCK_SIZE * WMMA_K];
    __shared__ half shmem_b[WMMA_K * BLOCK_SIZE];
    
    int blockRow = blockIdx.x * BLOCK_SIZE;
    int blockCol = blockIdx.y * BLOCK_SIZE;
    
    int warpRow = (threadIdx.x / warpSize) * WMMA_M;
    int warpCol = (threadIdx.x % warpSize / 4) * WMMA_N;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int i = 0; i < k; i += WMMA_K) {
        // Load block tiles into shared memory
        for (int j = threadIdx.x; j < BLOCK_SIZE * WMMA_K; j += blockDim.x) {
            int row = j / WMMA_K;
            int col = j % WMMA_K;
            if (blockRow + row < m && i + col < k) {
                shmem_a[j] = a[(blockRow + row) * k + (i + col)];
            }
        }
        
        for (int j = threadIdx.x; j < WMMA_K * BLOCK_SIZE; j += blockDim.x) {
            int row = j / BLOCK_SIZE;
            int col = j % BLOCK_SIZE;
            if (i + row < k && blockCol + col < n) {
                shmem_b[j] = b[(i + row) * n + (blockCol + col)];
            }
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, shmem_a + warpRow * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(b_frag, shmem_b + warpCol, BLOCK_SIZE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    int cRow = blockRow + warpRow;
    int cCol = blockCol + warpCol;
    
    if (cRow < m && cCol < n) {
        wmma::store_matrix_sync(c + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
    }
}

extern "C" {
    void launch_wmma_opt2(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream = 0) {
        dim3 gridDim((m + 63) / 64, (n + 63) / 64, 1);
        dim3 blockDim(256, 1, 1);
        wmma_opt2_block_tiling<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    }
}