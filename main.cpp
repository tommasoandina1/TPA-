#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>

using namespace std;

template<typename T>
class Matrix {
private:
    vector<vector<T>> data;
    int rows, cols;

public:
    Matrix(int r = 0, int c = 0) : rows(r), cols(c), data(r, vector<T>(c, 0)) {}

    int getRows() const { return rows; }
    int getColumns() const { return cols; }
    T getValue(int r, int c) const { return data[r][c]; }
    void setValue(int r, int c, T value) { data[r][c] = value; }

    bool readFromFile(const string& filename);
    void writeToFile(const string& filename) const;
    Matrix<T> convolution(const Matrix<T>& kernel) const;
    void SetMatrix(int r, int c, T value) { data[r][c] = value; }
};

template<typename T>
bool Matrix<T>::readFromFile(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Could not open file " << filename << endl;
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

template<typename T>
void Matrix<T>::writeToFile(const string& filename) const {
    ofstream file(filename);
    if (!file) {
        cerr << "Could not open file " << filename << endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

template<typename T>
Matrix<T> Matrix<T>::convolution(const Matrix<T>& kernel) const {
    int outputRows = rows - kernel.rows + 1;
    int outputCols = cols - kernel.cols + 1;
    Matrix<T> output(outputRows, outputCols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            T sum = 0;
            for (int ki = 0; ki < kernel.rows; ++ki) {
                for (int kj = 0; kj < kernel.cols; ++kj) {
                    sum += this->getValue(i + ki, j + kj) * kernel.getValue(ki, kj);
                }
            }
            output.setValue(i, j, sum);
        }
    }

    return output;
}

template <typename T>
Matrix<T> convolution_with_even_kernel_padding(const Matrix<T>& input, const Matrix<T>& kernel) {
    int kernel_rows = kernel.getRows();
    int kernel_columns = kernel.getColumns();
    int padding_row = kernel_rows / 2;
    int padding_col = kernel_columns / 2;

    int input_rows = input.getRows();
    int input_columns = input.getColumns();

    int output_rows = input_rows;
    int output_columns = input_columns;

    Matrix<T> output(output_rows, output_columns);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_columns; ++j) {
            T sum = 0;
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_columns; ++kj) {
                    int ii = i + ki - padding_row;
                    int jj = j + kj - padding_col;
                    if (ii >= 0 && ii < input_rows && jj >= 0 && jj < input_columns) {
                        sum += input.Get_Value(ii, jj) * kernel.Get_Value(ki, kj);
                    }
                }
            }
            output.Set_Matrix(i, j, sum);
        }
    }

    return output;
}



int main() {
        int inputSize = 100;
        int kernelSize = 4;

        Matrix<float> input(inputSize, inputSize);
        Matrix<float> kernel(kernelSize, kernelSize);
        Matrix<float> output(inputSize - kernelSize + 1, inputSize - kernelSize + 1);
        
        
        Matrix<float> padding(inputSize, inputSize);
        Matrix<float> output_padding(inputSize + 2 * 1 - kernelSize + 1, inputSize + 2 * 1 - kernelSize + 1);
        output_padding = convolution_parallel_with_padding(input, kernel, 1);
        output_padding.writeToFile("output_padding.txt");
        double start_p, end_p;
        start_p = omp_get_wtime();
        for (size_t i = 0; i < 100; i++)
        {
        #pragma omp parallel sections
        {
            #pragma omp section
       
            {
                input.readFromFile("matrix.txt");
            }
            #pragma omp section
            {
               kernel.readFromFile("kernel.txt");
            }
        }

        output = input.convolution(kernel);
        output.writeToFile("output_matrix.txt");

    } 
        
    end_p = omp_get_wtime();
    printf("Tempo parallelo: %g\n", (end_p - start_p) / 100);
    return 0;

}