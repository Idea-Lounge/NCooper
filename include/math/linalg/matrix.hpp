/*
    Copyright IdeaLounge.io 2018
*/

#ifndef NCOOPER_MATRIX_HPP_
#define NCOOPER_MATRIX_HPP_

#include <vector>
#include <iostream>
#include "vector.hpp"
#include <assert.h>

namespace ncooper {
    namespace math {
        namespace linalg {
            template <class MatrixType>
            class Matrix {
                public:
                    Matrix(); // initializes empty matrix
                    Matrix(int rows, int cols);
                    Matrix(int rows, int cols, MatrixType initVal);
                    Matrix(const Matrix<MatrixType>& matrix);
                    ~Matrix();

                    void push_back(Vector<MatrixType>& vector);
                    Matrix<MatrixType> transpose();
                    Matrix<MatrixType> hadamardProduct(const Matrix<MatrixType>& matrix);
                    Matrix<MatrixType> kroneckerProduct(const Matrix<MatrixType>& matrix);
                    Matrix<MatrixType> concatRight(const Matrix<MatrixType>& matrix);

                    int getRows() const;
                    int getCols() const;

                    MatrixType& operator()(int row, int col) {  // element access
                        return this->data[row][col];
                    }

                    const MatrixType& operator()(int row, int col) const {  // element access
                        return this->data[row][col];
                    }

                    const Vector<MatrixType>& operator[](int row) const {  // row access
                        return this->data[row];
                    }

                    Vector<MatrixType>& operator[](int row) {  // row access
                        return this->data[row];
                    }

                    // matrix1 + matrix2
                    friend Matrix<MatrixType> operator+(const Matrix<MatrixType>& matrix1, const Matrix<MatrixType>& matrix2) {
                        assert(matrix1.getCols() == matrix2.getCols()
                            && matrix1.getRows() == matrix2.getRows());
                        Matrix<MatrixType> finalMatrix(matrix1.getRows(), matrix1.getCols());
                        for (int i = 0; i < matrix1.getRows(); i++) {
                            finalMatrix[i] = matrix1[i] + matrix2[i];
                        }
                        return finalMatrix;  // would it not be more effecient if we pass by reference.
                    }

                    // scalar * matrix
                    friend Matrix<MatrixType> operator*(const MatrixType scalar, const Matrix<MatrixType>& matrix) {
                        Matrix<MatrixType> finalMatrix(matrix.getRows(), matrix.getCols(), 0);
                        for (int i = 0; i < matrix.getRows(); i++) {
                            finalMatrix[i] = scalar * matrix[i];
                        }
                        return finalMatrix;  // would it not be more effecient if we pass by reference.
                    }

                    // matrix * vector
                    friend Vector<MatrixType> operator*(const Matrix<MatrixType>& matrix, const Vector<MatrixType>& vector) {
                        assert(matrix.getCols() == vector.getSize());
                        Vector<MatrixType> finalVector(matrix.getRows(), 0);
                        for (int i = 0; i < matrix.getRows(); i++) {
                            finalVector[i] = matrix[i] * vector;
                        }
                        return finalVector;  // would it not be more effecient if we pass by reference.
                    }

                    // matrix1 * matrix2
                    friend Matrix<MatrixType> operator*(const Matrix<MatrixType>& matrix1, const Matrix<MatrixType>& matrix2) {
                        assert(matrix1.getCols() == matrix2.getRows());
                        Matrix<MatrixType> finalMatrix(matrix1.getRows(), matrix2.getCols(), 0);
                        for (int i = 0; i < matrix1.getRows(); i++) {
                            for (int j = 0; j < matrix2.getCols(); j++) {
                                for (int k = 0; k < matrix1.getCols(); k++) {
                                    finalMatrix(i, j) += matrix1(i, k) * matrix2(k, j);
                                }
                            }
                        }
                        return finalMatrix;  // would it not be more effecient if we pass by reference.
                    }

                    friend std::ostream& operator<<(std::ostream& os, const Matrix<MatrixType>& matrix) {
                		for (int i = 0; i < matrix.getRows(); i++){
                			os << matrix[i] << '\n';
                		}
                	    return os;
                	}
                protected:
                    std::vector<Vector<MatrixType>> data;
                    int rows = 0;
                    int cols = 0;
            };
        }
    }
}

#endif  // NCOOPER_MATRIX_HPP_
