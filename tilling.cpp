#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <string>  // Include for std::string

class Matrix {
public:
    std::vector<std::vector<int>> data;
    int rows, cols;

    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<int>(c, 0)) {}

    std::vector<int>& operator[](int i) {
        return data[i];
    }

    const std::vector<int>& operator[](int i) const {
        return data[i];
    }

    bool readFromFile(const std::string& filename);
    void writeToFile(const std::string& filename) const;
};

bool Matrix::readFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Could not open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> data[i][j];
        }
    }
    file.close();
    return true;
}

void Matrix::writeToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i][j] << " ";
        }
        file << std::endl;
    }
    file.close();
}

void Tilling(const Matrix& input, const Matrix& kernel, Matrix& output) {
    int output_rows = input.rows - kernel.rows + 1;
    int output_cols = input.cols - kernel.cols + 1;
    int kernel_rows = kernel.rows;
    int kernel_cols = kernel.cols;

    const int BLOCK_SIZE = 64; // dimensione cache memoria L1

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < output_rows; block_i += BLOCK_SIZE) {
        for (int block_j = 0; block_j < output_cols; block_j += BLOCK_SIZE) {
            for (int i = block_i; i < std::min(block_i + BLOCK_SIZE, output_rows); ++i) {
                for (int j = block_j; j < std::min(block_j + BLOCK_SIZE, output_cols); ++j) {
                    int sum = 0;
                    for (int ki = 0; ki < kernel_rows; ++ki) {
                        for (int kj = 0; kj < kernel_cols; ++kj) {
                            sum += input[i + ki][j + kj] * kernel[ki][kj];
                        }
                    }
                    output[i][j] = sum;
                }
            }
        }
    }
}

int main() {
    int inputSize = 1000;
    int kernelSize = 4;

    Matrix input(inputSize, inputSize);
    Matrix kernel(kernelSize, kernelSize);
    Matrix output(inputSize - kernelSize + 1, inputSize - kernelSize + 1);

    double start_p, end_p;
    start_p = omp_get_wtime();

    for (size_t i = 0; i < 100; i++) {
        input.readFromFile("matrix.txt");
        kernel.readFromFile("kernel.txt");
        Tilling(input, kernel, output);
        output.writeToFile("output.txt");
    }

    end_p = omp_get_wtime();
    printf("Parallel time: %g\n", (end_p - start_p) / 100);

    return 0;
}
