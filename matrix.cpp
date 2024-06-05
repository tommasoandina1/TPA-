#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>
using namespace std;

// Classe matrice
template <typename T>
class Matrix {
private:
    vector<vector<T> > data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<T>(c)) {}

    int getRows() const { return rows; }
    int getColumns() const { return cols; }
    T Get_Value(int r, int c) const { return data[r][c]; }
    void Set_Matrix(int r, int c, T value) { data[r][c] = value; }

    bool readFromFile(const string &filename) {
        ifstream file(filename.c_str());
        if (!file.is_open()) {
            cerr << "Could not open file " << filename << endl;
            return false;
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (!(file >> data[i][j])) {
                    cerr << "Invalid data in file " << filename << endl;
                    return false;
                }
            }
        }

        file.close();
        return true;
    }

    bool writeToFile(const string &filename) const {
        ofstream file(filename.c_str());
        if (!file.is_open()) {
            cerr << "Could not open file " << filename << endl;
            return false;
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << data[i][j] << " ";
            }
            file << endl;
        }

        file.close();
        return true;
    }

    void printToFile(const string &filename) const {
        ofstream file(filename.c_str());
        if (!file.is_open()) {
            cerr << "Could not open file " << filename << endl;
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file << data[i][j] << " ";
            }
            file << endl;
        }

        file.close();
    }
};

// Funzione per calcolare la convoluzione parallelizzata
template <typename T>
Matrix<T> convolution_parallel(const Matrix<T>& input, const Matrix<T>& kernel) {
    int kernel_rows = kernel.getRows();
    int kernel_columns = kernel.getColumns();

    int input_rows = input.getRows();
    int input_columns = input.getColumns();

    int output_rows = input_rows - kernel_rows + 1;
    int output_columns = input_columns - kernel_columns + 1;

    if (output_rows <= 0 || output_columns <= 0) {
        throw invalid_argument("Kernel size is larger than input size.");
    }

    Matrix<T> output(output_rows, output_columns);

    int num_threads = 3;
    int row_block_size = input_rows / 5;
    int column_block_size = input_columns / 2;

    // Gestione del resto
    int extra_rows = input_rows % 5;
    int extra_columns = input_columns % 2;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int start_row = (thread_id / 2) * row_block_size;
        int end_row = start_row + row_block_size + (thread_id / 2 < extra_rows ? 1 : 0);
        int start_col = (thread_id % 2) * column_block_size;
        int end_col = start_col + column_block_size + (thread_id % 2 < extra_columns ? 1 : 0);

        // Sovrapposizione per convoluzione
        if (start_col > 0) start_col--;
        if (end_col < input_columns) end_col++;

        for (int i = start_row; i < end_row; ++i) {
            for (int j = start_col; j < end_col; ++j) {
                T sum = 0;
                for (int ki = 0; ki < kernel_rows; ++ki) {
                    for (int kj = 0; kj < kernel_columns; ++kj) {
                        sum += input.Get_Value(i + ki, j + kj) * kernel.Get_Value(ki, kj);
                    }
                }
                #pragma omp critical
                {
                    output.Set_Matrix(i - start_row, j - start_col, sum);
                }
            }
        }
    }

    return output;
}

int main() {
    // Leggi la matrice di input e il kernel dai file
    int input_size = 10; // Sostituire con la dimensione effettiva della matrice di input
    Matrix<double> input(input_size, input_size);  // Matrice di input quadrata
    Matrix<double> kernel(4, 4); // Kernel 4x4

    if (!input.readFromFile("matrix.txt")) {
        return EXIT_FAILURE;
    }
    if (!kernel.readFromFile("kernel.txt")) {
        return EXIT_FAILURE;
    }

    cout << "Input Matrix:" << endl;
    input.printToFile("input_matrix.txt");

    cout << "Kernel Matrix:" << endl;
    kernel.printToFile("kernel_matrix.txt");

    // Esegui la convoluzione parallelizzata
    Matrix<double> output_parallel = convolution_parallel(input, kernel);
    cout << "Output Matrix (Parallel):" << endl;
    output_parallel.printToFile("output_parallel.txt");

    return EXIT_SUCCESS;
}


