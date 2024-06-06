#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <algorithm> 
#include <cstdio> 

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

    static Matrix<T> randomMatrix(int rows, int cols) {
        srand(time(0));
        Matrix<T> random(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                random.data[i][j] = static_cast<T>(rand() % 100); // Numero casuale tra 0 e 99
            }
        }
        return random;
    }

    bool isEqual(const Matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols) {
            return false; // Le dimensioni sono diverse
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (data[i][j] != other.data[i][j]) {
                    return false; // I valori sono diversi
                }
            }
        }

        return true; // Le matrici sono uguali
    }
};


template <typename T>
Matrix<T> convolution_sequential(const Matrix<T>& input, const Matrix<T>& kernel) {
    int kernel_rows = kernel.getRows();
    int kernel_columns = kernel.getColumns();

    int input_rows = input.getRows();
    int input_columns = input.getColumns();

    int output_rows = input_rows - kernel_rows + 1;
    int output_columns = input_columns - kernel_columns + 1;

    if (output_rows <= 0 || output_columns <= 0) {
        throw std::invalid_argument("Il kernel è maggiore della dimensione dell'input.");
    }

    Matrix<T> output(output_rows, output_columns);

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_columns; ++j) {
            T sum = 0;
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_columns; ++kj) {
                    sum += input.Get_Value(i + ki, j + kj) * kernel.Get_Value(ki, kj);
                }
            }
            output.Set_Matrix(i, j, sum);
        }
    }

    return output;
}

template <typename T>
Matrix<T> convolution_parallel(const Matrix<T>& input, const Matrix<T>& kernel) {
    int kernel_rows = kernel.getRows();
    int kernel_columns = kernel.getColumns();

    int input_rows = input.getRows();
    int input_columns = input.getColumns();

    int output_rows = input_rows - kernel_rows + 1;
    int output_columns = input_columns - kernel_columns + 1;

    Matrix<T> output(output_rows, output_columns);

    int num_threads = 11;
    omp_set_num_threads(num_threads);

    #pragma omp parallel shared(input, kernel, output)
    {
        int num_threads_actual = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        //printf("Thread %d su %d thread totali sta lavorando\n", thread_id, num_threads_actual);

        int rows_per_thread = (output_rows + num_threads_actual - 1) / num_threads_actual; // Calcola il numero di righe per thread, arrotondato per eccesso
        int start_row = thread_id * rows_per_thread;
        int end_row = min(start_row + rows_per_thread, output_rows);

        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < output_columns; ++j) {
                T sum = 0;
                for (int ki = 0; ki < kernel_rows; ++ki) {
                    for (int kj = 0; kj < kernel_columns; ++kj) {
                        sum += input.Get_Value(i + ki, j + kj) * kernel.Get_Value(ki, kj);
                    }
                }
                output.Set_Matrix(i, j, sum);
                //printf("Thread %d sta calcolando output[%d][%d]\n", thread_id, i, j);
            }
        }
    }

    return output;
}



int main() {
    int input_size = 1000; 
    Matrix<double> input = Matrix<double>::randomMatrix(input_size, input_size);
    Matrix<double> kernel(4, 4); // Kernel 4x4
    input.printToFile("matrix.txt");

    //Sequenziale 
    double start_s, end_s;
    start_s = omp_get_wtime();

    for (size_t i = 0; i < 100; i++)
    {
        Matrix<double> input_seq(input_size, input_size);
        Matrix<double> kernel_seq(4, 4); 
        input_seq.readFromFile("matrix.txt");
        kernel_seq.readFromFile("kernel.txt");
        Matrix<double> output_sequential = convolution_sequential(input_seq, kernel_seq);
        output_sequential.printToFile("output_seq.txt");
        
    }
    
    end_s = omp_get_wtime();
    printf("Tempo sequenziale: %g\n", (end_s-start_s)/input_size);

    // Parallelo
    double start_p, end_p;
    start_p = omp_get_wtime();

    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        Matrix<double> input_par(input_size, input_size);
        Matrix<double> kernel_par(4, 4); 
        input_par.readFromFile("matrix.txt");
        kernel_par.readFromFile("kernel.txt");

        #pragma omp for
        for (size_t i = 0; i < 100; i++)
        {
        Matrix<double> output_par = convolution_parallel(input_par, kernel_par); // Utilizza la versione parallela
        output_par.printToFile("output_par.txt");
        }
    }

    end_p = omp_get_wtime();

printf("Tempo Parallelo: %g\n", (end_p-start_p)/input_size);
    // Verifica che i risultati siano uguali
    
    Matrix<double> output_seq(input_size, input_size);
    output_seq.readFromFile("output_seq.txt");
    Matrix<double> output_par(input_size, input_size);
    output_par.readFromFile("output_par.txt");

    if (output_seq.isEqual(output_par)) {

        cout << "I risultati sono uguali" << endl;
    } else {
        cout << "I risultati sono diversi" << endl;
    }
    return EXIT_SUCCESS;
}
