#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cufft.h>

using namespace std;

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256

// Use std::complex for CPU and thrust::complex for GPU
using CPUComplex = complex<double>;
using GPUComplex = thrust::complex<double>;

// Random Number Generator 
template <typename ComplexType>
vector<ComplexType> generateRandomComplexVector(int N, int seed = 42, double minVal = -10.0, double maxVal = 10.0) 
{
    mt19937 gen(seed);
    uniform_real_distribution<> dis(minVal, maxVal);
    
    vector<ComplexType> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = ComplexType(dis(gen), dis(gen));
    }
    
    return input;
}

// CPU Bit Reversal Function
vector<CPUComplex> cpuBitReverse(const vector<CPUComplex>& input) {
    int N = input.size();
    int logN = 0;
    while ((1 << logN) < N) logN++;

    vector<CPUComplex> output(N);
    for (int i = 0; i < N; i++) {
        int reversed = 0;
        for (int j = 0; j < logN; j++) {
            reversed = (reversed << 1) | ((i >> j) & 1);
        }
        output[i] = input[reversed];
    }
    return output;
}

// CPU Sequential FFT Implementation
vector<CPUComplex> sequentialFFT(const vector<CPUComplex>& input) {
    int N = input.size();
    if (N & (N - 1)) {
        throw invalid_argument("Input size must be a power of 2");
    }

    vector<CPUComplex> data = cpuBitReverse(input);

    int logN = log2(N);
    for (int stage = 0; stage < logN; stage++) {
        int groupSize = 1 << stage;
        int butterfliesPerGroup = N / (2 * groupSize);

        for (int group = 0; group < butterfliesPerGroup; group++) {
            for (int pos = 0; pos < groupSize; pos++) {
                // Twiddle Factor Calculation
                double angle = -2.0 * M_PI * pos / (2 * groupSize);
                CPUComplex twiddle = CPUComplex(cos(angle), sin(angle));
                // Even Odd index split
                int evenIdx = 2 * group * groupSize + pos;
                int oddIdx = evenIdx + groupSize;

                CPUComplex even = data[evenIdx];
                CPUComplex odd = data[oddIdx] * twiddle;

                data[evenIdx] = even + odd;
                data[oddIdx] = even - odd;
            }
        }
    }

    return data;
}

// GPU Bit Reversal Kernel
__global__ void bitReversalKernel(GPUComplex* data, int N, int logN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int reversed = 0;
        for (int j = 0; j < logN; j++) {
            reversed = (reversed << 1) | ((idx >> j) & 1);
        }
       
        if (idx < reversed) {
            GPUComplex temp = data[idx];
            data[idx] = data[reversed];
            data[reversed] = temp;
        }
    }
}

// GPU Butterfly Kernel
__global__ void butterflyKernel(GPUComplex* data, int N, int stage) {
    int groupSize = 1 << stage;
    int butterfliesPerGroup = N / (2 * groupSize);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < butterfliesPerGroup * groupSize) {
        int group = idx / groupSize;
        int pos = idx % groupSize;
        
        double angle = -2.0 * M_PI * pos / (2.0 * groupSize);
        GPUComplex twiddle = GPUComplex(cos(angle), sin(angle));
        
        int evenIdx = 2 * group * groupSize + pos;
        int oddIdx = evenIdx + groupSize;
        
        GPUComplex even = data[evenIdx];
        GPUComplex odd = data[oddIdx] * twiddle;
        
        data[evenIdx] = even + odd;
        data[oddIdx] = even - odd;
    }
}

// CUDA FFT Implementation
vector<GPUComplex> cudaFFT(const vector<GPUComplex>& input) {
    int N = input.size();
    
    if (N & (N - 1)) {
        throw invalid_argument("Input size must be a power of 2");
    }
    
    int logN = 0;
    while ((1 << logN) < N) logN++;
    size_t complexSize = sizeof(GPUComplex);
    
    GPUComplex* unified_data;
    cudaMallocManaged(&unified_data, N * complexSize);
    
    for (int i = 0; i < N; i++) {
        unified_data[i] = input[i];
    }
    
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(unified_data, N * complexSize, device);
    
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bitReversalKernel<<<numBlocks, THREADS_PER_BLOCK>>>(unified_data, N, logN);
    cudaDeviceSynchronize();
    
    for (int stage = 0; stage < logN; stage++) {
        int groupSize = 1 << stage;
        int butterfliesPerGroup = N / (2 * groupSize);
        int totalThreads = butterfliesPerGroup * groupSize;
        numBlocks = (totalThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        butterflyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(unified_data, N, stage);
        cudaDeviceSynchronize();
    }
    
    cudaMemPrefetchAsync(unified_data, N * complexSize, cudaCpuDeviceId);
    cudaDeviceSynchronize();
    
    vector<GPUComplex> output(unified_data, unified_data + N);
    
    cudaFree(unified_data);
    
    return output;
}

// Function to print complex array
template <typename ComplexType>
void printComplexArray(const vector<ComplexType>& arr, int printCount = 10) {
    printCount = min(printCount, static_cast<int>(arr.size()));
    for (int i = 0; i < printCount; i++) {
        cout << fixed << setprecision(4) 
             << "(" << arr[i].real() << " + " << arr[i].imag() << "i) ";
    }
    cout << endl;
}

// Function to verify FFT results against cuFFT
bool verifyCuFFT(const vector<GPUComplex>& input) {
    size_t N = input.size();
    
    GPUComplex* d_input = nullptr;
    GPUComplex* d_output = nullptr;
    cudaMallocManaged(&d_input, N * sizeof(GPUComplex));
    cudaMallocManaged(&d_output, N * sizeof(GPUComplex));
    
    
    for (size_t i = 0; i < N; ++i) {
        d_input[i] = input[i];
    }
    
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_input, N * sizeof(GPUComplex), device);
    cudaMemPrefetchAsync(d_output, N * sizeof(GPUComplex), device);
    
    // Create cuFFT plan
    cufftHandle plan;
    // Cast N to int for cufft API
    cufftPlan1d(&plan, static_cast<int>(N), CUFFT_Z2Z, 1);
    
    // Execute forward FFT
    cufftExecZ2Z(plan, 
        reinterpret_cast<cufftDoubleComplex*>(d_input), 
        reinterpret_cast<cufftDoubleComplex*>(d_output), 
        CUFFT_FORWARD);
    
    
    vector<GPUComplex> cuFFTOutput(N);
    
    // Prefetch results back to CPU
    cudaMemPrefetchAsync(d_output, cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();
    

    for (size_t i = 0; i < N; ++i) {
        cuFFTOutput[i] = d_output[i];
    }
    

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
    

    vector<GPUComplex> ourCudaOutput = cudaFFT(input);
    
    // Check results
    for (size_t i = 0; i < N; ++i) {
        double diff_real = abs(ourCudaOutput[i].real() - cuFFTOutput[i].real());
        double diff_imag = abs(ourCudaOutput[i].imag() - cuFFTOutput[i].imag());
        
        if (diff_real > 1e-6 || diff_imag > 1e-6) {
            cout << "Verification Failed: Mismatch at index " << i 
                 << "\nOur output: (" << ourCudaOutput[i].real() << " + " << ourCudaOutput[i].imag() << "i)"
                 << "\ncuFFT output: (" << cuFFTOutput[i].real() << " + " << cuFFTOutput[i].imag() << "i)"
                 << "\nDiff_Real: " << diff_real << ", Diff_Imag: " << diff_imag << endl;
            return false;
        }
    }
    
    cout << "cuFFT Verification Passed" << endl;
    return true;
}

int main() {
    try {
        const int N = 1048576;  // 2^20
        const int SEED = 42;

        // Generate input
        vector<CPUComplex> cpuInput = generateRandomComplexVector<CPUComplex>(N, SEED);
        
        // Convert CPU input to GPU complex for CUDA
        vector<GPUComplex> gpuInput(N);
        transform(cpuInput.begin(), cpuInput.end(), gpuInput.begin(), 
            [](const CPUComplex& c) { return GPUComplex(c.real(), c.imag()); });
        
        // CPU Sequential FFT
        auto cpuStart = chrono::high_resolution_clock::now();
        vector<CPUComplex> cpuOutput = sequentialFFT(cpuInput);
        auto cpuEnd = chrono::high_resolution_clock::now();
        
        chrono::duration<double> cpuElapsed = cpuEnd - cpuStart;
        cout << "Time taken for Sequential FFT: " << cpuElapsed.count() << " seconds\n";
        
        // CUDA FFT
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        vector<GPUComplex> gpuOutput = cudaFFT(gpuInput);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cout << "Time taken for CUDA FFT: " << milliseconds / 1000.0 << " seconds\n";
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Verify results with cuFFT
        bool cuFFTVerificationPassed = verifyCuFFT(gpuInput);
        
        // Print results
        cout << "\nFirst 10 CPU Input Values:\n";
        printComplexArray(cpuInput);
        cout << "First 10 CPU FFT Output Values:\n";
        printComplexArray(cpuOutput);
        
        cout << "\nFirst 10 GPU Input Values:\n";
        printComplexArray(gpuInput);
        cout << "First 10 GPU FFT Output Values:\n";
        printComplexArray(gpuOutput);
        
        return cuFFTVerificationPassed ? 0 : 1;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}