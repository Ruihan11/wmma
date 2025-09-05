#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Optimization Level 1: Shared Memory + Double Buffering
__global__ void wmma_opt1_shared_memory(
    half* a, half* b, float* c,
    int m, int n, int k
) {
    __shared__ half shmem_a[WMMA_M * WMMA_K * 2];
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        if (aRow < m && aCol < k && bRow < k && bCol < n) {
            // Use shared memory for better cache locality
            int shmem_idx = threadIdx.x % 2;
            
            // Copy to shared memory
            if (threadIdx.x < WMMA_M * WMMA_K / 32) {
                for (int j = 0; j < 32; j++) {
                    int idx = threadIdx.x * 32 + j;
                    if (idx < WMMA_M * WMMA_K) {
                        shmem_a[shmem_idx * WMMA_M * WMMA_K + idx] = a[aRow * k + aCol + idx];
                    }
                }
            }
            
            __syncthreads();
            
            wmma::load_matrix_sync(a_frag, &shmem_a[shmem_idx * WMMA_M * WMMA_K], WMMA_K);
            wmma::load_matrix_sync(b_frag, b + bRow * n + bCol, n);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < m && cCol < n) {
        wmma::store_matrix_sync(c + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
    }
}

extern "C" {
    void launch_wmma_opt1(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream = 0) {
        dim3 gridDim((m + 15) / 16, (n + 15) / 16, 1);
        dim3 blockDim(32, 1, 1);
        wmma_opt1_shared_memory<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    }
}