#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <complex>
#include <stdexcept>
#include <iomanip>
#ifndef M_PI
    #define M_PI (std::acos(-1))
#endif

// Complex number type alias for clarity
using Complex = std::complex<double>;

// Bit reversal function for reordering
std::vector<Complex> bitReverse(const std::vector<Complex>& input) {
    int N = input.size();
    int logN = 0;
    while ((1 << logN) < N) logN++;

    std::vector<Complex> output(N);
    for (int i = 0; i < N; i++) {
        // Compute bit-reversed index
        int reversed = 0;
        for (int j = 0; j < logN; j++) {
            reversed = (reversed << 1) | ((i >> j) & 1);
        }
        output[i] = input[reversed];
    }
    return output;
}

// Sequential FFT implementation
std::vector<Complex> sequentialFFT(const std::vector<Complex>& input) {
    // Ensure input size is a power of 2
    int N = input.size();
    if (N & (N - 1)) {
        throw std::invalid_argument("Input size must be a power of 2");
    }

    // Bit reversal stage
    std::vector<Complex> data = bitReverse(input);

    // FFT stages
    int logN = std::log2(N);
    for (int stage = 0; stage < logN; stage++) {
        int groupSize = 1 << stage;
        int butterfliesPerGroup = N / (2 * groupSize);

        for (int group = 0; group < butterfliesPerGroup; group++) {
            for (int pos = 0; pos < groupSize; pos++) {
                // Compute twiddle factor
                double angle = -2.0 * M_PI * pos / (2 * groupSize);
                Complex twiddle = std::polar(1.0, angle);

                // Butterfly indices
                int evenIdx = 2 * group * groupSize + pos;
                int oddIdx = evenIdx + groupSize;

                // Butterfly computation
                Complex even = data[evenIdx];
                Complex odd = data[oddIdx] * twiddle;

                data[evenIdx] = even + odd;
                data[oddIdx] = even - odd;
            }
        }
    }

    return data;
}

// Utility function to print complex array
void printComplexArray(const std::vector<Complex>& arr, int printCount = 10) {
    printCount = std::min(printCount, static_cast<int>(arr.size()));
    for (int i = 0; i < printCount; i++) {
        std::cout << std::fixed << std::setprecision(4) 
                  << "(" << arr[i].real() << " + " << arr[i].imag() << "i) ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        // Set up parameters
        const int N = 8192;  // Power of 2
        
        // Generate input data matching sequential version
        std::vector<Complex> input(N);
        input[0] = Complex(8.3068, 4.5250);
        input[1] = Complex(0.3072, 1.8865);
        input[2] = Complex(3.2312, 9.5030);
        input[3] = Complex(5.7699, 0.5730);
        input[4] = Complex(-3.4030, -8.5180);
        input[5] = Complex(-9.3806, 1.6749);
        input[6] = Complex(-9.6050, 8.5719);
        input[7] = Complex(-3.2791, 5.5874);
        input[8] = Complex(-1.7036, -3.1495);
        input[9] = Complex(3.0296, 8.1463);
        
        // Initialize remaining values with fixed seed
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for (int i = 10; i < N; i++) {
            input[i] = Complex(dis(gen), dis(gen));
        }
        
        // Perform CUDA FFT
        std::vector<Complex> output = sequentialFFT(input);
        
        // Print results
        std::cout << "First 10 Input Values:\n";
        printComplexArray(input);
        
        std::cout << "\nFirst 10 FFT Output Values:\n";
        printComplexArray(output);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}