#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// cuBLAS wrapper kernel for performance comparison
class CuBLASWrapper {
private:
    static cublasHandle_t handle;
    static bool initialized;

public:
    static void initialize() {
        if (!initialized) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
            initialized = true;
        }
    }
    
    static void cleanup() {
        if (initialized) {
            cublasDestroy(handle);
            initialized = false;
        }
    }
    
    static void gemm(half* d_a, half* d_b, float* d_c, int m, int n, int k) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cublasGemmEx(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    d_b, CUDA_R_16F, n,
                    d_a, CUDA_R_16F, k,
                    &beta,
                    d_c, CUDA_R_32F, n,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    static void warmup(half* d_a, half* d_b, float* d_c, int m, int n, int k) {
        // Perform warmup runs
        for (int i = 0; i < 3; i++) {
            gemm(d_a, d_b, d_c, m, n, k);
        }
        cudaDeviceSynchronize();
    }
};

// Static member initialization
cublasHandle_t CuBLASWrapper::handle;
bool CuBLASWrapper::initialized = false;

extern "C" {
    void launch_wmma_opt0_cublas_init() {
        CuBLASWrapper::initialize();
    }
    
    void launch_wmma_opt0_cublas_cleanup() {
        CuBLASWrapper::cleanup();
    }
    
    void launch_wmma_opt0_cublas_warmup(half* d_a, half* d_b, float* d_c, int m, int n, int k) {
        CuBLASWrapper::warmup(d_a, d_b, d_c, m, n, k);
    }
    
    void launch_wmma_opt0_cublas(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream = 0) {
        CuBLASWrapper::gemm(d_a, d_b, d_c, m, n, k);
    }
}