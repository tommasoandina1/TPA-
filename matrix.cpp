#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

template <typename T>
Matrix<T>::Matrix(int rows, int columns) : rows(rows), columns(columns) {
    data.resize(rows, vector<T>(columns, 0));
}

template <typename T>
void Matrix<T>::Set_Matrix(int row, int column, T x) {
    if (row >= 0 && row < rows && column >= 0 && column < columns) {
        data[row][column] = x;
    } else {
        cout << "errore" << endl;
    }
}

template <typename T>
T Matrix<T>::Get_Value(int row, int column) const {
    if (row >= 0 && row < rows && column >= 0 && column < columns) {
        return data[row][column];
    } else {
        cout << "errore" << endl;
        return T();
    }
}

template <typename T>
void Matrix<T>::Print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << data[i][j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
void Matrix<T>::readFromTerminal() {
    cout << "Enter the elements of the matrix row by row:" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cin >> data[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::readFromFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Unable to open file" << endl;
        return;
    }

    string line;
    int row = 0;
    while (getline(file, line) && row < rows) {
        stringstream ss(line);
        for (int col = 0; col < columns; ++col) {
            ss >> data[row][col];
        }
        row++;
    }

    if (row < rows) {
        cout << "File contains less data than expected" << endl;
    }

    file.close();
}

template <typename T>

Matrix<T> Matrix<T>::random_identity_matrix() {
    srand(time(0));
    int x = rand() % 100 + 1; // Numero casuale tra 1 e 100
    Matrix identity(x, x);
    for (int i = 0; i < x; ++i) {
        identity.data[i][i] = 1.0;
    }
    return identity;
}

template <typename T>
Matrix<T> Matrix<T>::sum_matrix(const Matrix& other) const {
    if (rows != other.rows || columns != other.columns) {
        cout << "Le dimensioni delle matrici non corrispondono per la somma" << endl;
        return Matrix(0, 0);
    }

    Matrix result(rows, columns);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::dot_matrix(const Matrix& other) const {
    if (columns != other.rows) {
        cout << "errore di dimensioni" << endl;
        return Matrix(0, 0);
    }

    Matrix result(rows, other.columns);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.columns; ++j) {
            for (int k = 0; k < other.rows; ++k) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix result(columns, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}
