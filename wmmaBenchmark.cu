#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <iomanip>
#include <algorithm>
#include <fstream>

extern "C" {
    // Optimization levels
    void launch_wmma_opt0_cublas_init();
    void launch_wmma_opt0_cublas_cleanup();
    void launch_wmma_opt0_cublas_warmup(half* d_a, half* d_b, float* d_c, int m, int n, int k);
    void launch_wmma_opt0_cublas(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream);
    void launch_wmma_opt1(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream);
    void launch_wmma_opt2(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream);
    void launch_wmma_opt3(half* d_a, half* d_b, float* d_c, int m, int n, int k, cudaStream_t stream);
}

class WMMABenchmark {
private:
    int m, n, k;
    half *d_a, *d_b;
    float *d_c;
    std::vector<half> h_a, h_b;
    std::vector<float> h_c;
    cudaEvent_t start, stop;
    
public:
    WMMABenchmark(int m, int n, int k) : m(m), n(n), k(k) {
        // Check for reasonable matrix sizes to prevent crashes
        size_t total_elements = static_cast<size_t>(m) * n * k;
        if (total_elements > 1e10) { // Roughly 4GB limit for FP16
            throw std::runtime_error("Matrix too large - would exceed memory limits");
        }
        
        try {
            h_a.resize(m * k);
            h_b.resize(k * n);
            h_c.resize(m * n);
        } catch (const std::bad_alloc& e) {
            throw std::runtime_error("Failed to allocate host memory");
        }
        
        cudaError_t err;
        err = cudaMalloc(&d_a, m * k * sizeof(half));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for matrix A");
        }
        
        err = cudaMalloc(&d_b, k * n * sizeof(half));
        if (err != cudaSuccess) {
            cudaFree(d_a);
            throw std::runtime_error("Failed to allocate device memory for matrix B");
        }
        
        err = cudaMalloc(&d_c, m * n * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_a);
            cudaFree(d_b);
            throw std::runtime_error("Failed to allocate device memory for matrix C");
        }
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        launch_wmma_opt0_cublas_init();
        
        initializeData();
    }
    
    ~WMMABenchmark() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        launch_wmma_opt0_cublas_cleanup();
    }
    
    void initializeData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < m * k; i++) {
            h_a[i] = __float2half(dis(gen));
        }
        
        for (int i = 0; i < k * n; i++) {
            h_b[i] = __float2half(dis(gen));
        }
        
        for (int i = 0; i < m * n; i++) {
            h_c[i] = 0.0f;
        }
        
        cudaMemcpy(d_a, h_a.data(), m * k * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), k * n * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    
    
    float benchmarkWMMAOpt1(int iterations = 100) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            launch_wmma_opt1(d_a, d_b, d_c, m, n, k, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds / iterations;
    }
    
    float benchmarkWMMAOpt2(int iterations = 100) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            launch_wmma_opt2(d_a, d_b, d_c, m, n, k, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds / iterations;
    }
    
    
    float benchmarkWMMAOpt0CuBLAS(int iterations = 100) {
        // Warmup
        launch_wmma_opt0_cublas_warmup(d_a, d_b, d_c, m, n, k);
        
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            launch_wmma_opt0_cublas(d_a, d_b, d_c, m, n, k, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds / iterations;
    }
    
    float benchmarkWMMAOpt3(int iterations = 100) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            launch_wmma_opt3(d_a, d_b, d_c, m, n, k, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds / iterations;
    }
    
    
    void printResults(const std::string& kernelName, float avgTime) {
        double gflops = (2.0 * m * n * k) / (avgTime * 1e-3) / 1e9;
        std::cout << kernelName << ":" << std::endl;
        std::cout << "  Matrix size: " << m << "x" << k << " * " << k << "x" << n << std::endl;
        std::cout << "  Average time: " << avgTime << " ms" << std::endl;
        std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << std::endl;
    }
    
    void outputCSV(const std::vector<std::pair<std::string, float>>& results, const std::string& filename = "benchmark_results.csv") {
        std::ofstream csvFile(filename, std::ios::app);
        if (!csvFile.is_open()) {
            std::cerr << "Error opening CSV file: " << filename << std::endl;
            return;
        }
        
        for (const auto& result : results) {
            if (result.second > 0) {
                double gflops = (2.0 * m * n * k) / (result.second * 1e-3) / 1e9;
                csvFile << m << "," << n << "," << k << "," 
                        << result.first << "," 
                        << result.second << "," 
                        << gflops << std::endl;
            }
        }
        
        csvFile.close();
    }
    
    void runAllBenchmarks(int iterations = 100) {
        std::cout << "=== WMMA Benchmark Results ===" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "GPU: ";
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << prop.name << std::endl << std::endl;
        
        // Store results for sorting
        std::vector<std::pair<std::string, float>> results;
        
        // Benchmark cuBLAS first as the gold standard
        float opt0Time = benchmarkWMMAOpt0CuBLAS(iterations);
        results.push_back({"WMMA Opt0 cuBLAS", opt0Time});
        
        // Optimization kernels
        float opt1Time = benchmarkWMMAOpt1(iterations);
        results.push_back({"WMMA Opt1 Shared Memory", opt1Time});
        
        float opt2Time = benchmarkWMMAOpt2(iterations);
        results.push_back({"WMMA Opt2 Block Tiling", opt2Time});
        
        float opt3Time = benchmarkWMMAOpt3(iterations);
        results.push_back({"WMMA Opt3 Optimized", opt3Time});
        
        // Sort by GFLOPS (descending)
        std::sort(results.begin(), results.end(), [this](const auto& a, const auto& b) {
            if (a.second <= 0 || b.second <= 0) return false; // Handle invalid times
            double gflops_a = (2.0 * m * n * k) / (a.second * 1e-3) / 1e9;
            double gflops_b = (2.0 * m * n * k) / (b.second * 1e-3) / 1e9;
            return gflops_a > gflops_b;
        });
        
        // Print results sorted by performance
        for (const auto& result : results) {
            if (result.second > 0) { // Only print valid results
                printResults(result.first, result.second);
            } else {
                std::cout << result.first << ": FAILED" << std::endl;
            }
        }
        
        // Output to CSV
        outputCSV(results);
    }
    
};

int main(int argc, char* argv[]) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 7) {
        std::cerr << "WMMA requires compute capability 7.0 or higher" << std::endl;
        return -1;
    }
    
    // Initialize CSV file with headers
    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
        csvFile << "M,N,K,Kernel,Time_ms,GFLOPS" << std::endl;
        csvFile.close();
    }
    
    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048}
    };
    
    int iterations = 100;
    
    if (argc >= 4) {
        int m = std::atoi(argv[1]);
        int n = std::atoi(argv[2]);
        int k = std::atoi(argv[3]);
        sizes = {{m, n, k}};
        
        if (argc >= 5) {
            iterations = std::atoi(argv[4]);
        }
    }
    
    for (auto [m, n, k] : sizes) {
        std::cout << "Testing matrix size: " << m << "x" << n << "x" << k << std::endl;
        
        try {
            WMMABenchmark benchmark(m, n, k);
            benchmark.runAllBenchmarks(iterations);
        } catch (const std::exception& e) {
            std::cerr << "Error with matrix size " << m << "x" << n << "x" << k 
                      << ": " << e.what() << std::endl;
            std::cout << "Skipping this matrix size..." << std::endl;
        }
        
        std::cout << "=========================" << std::endl << std::endl;
    }
    
    return 0;
}