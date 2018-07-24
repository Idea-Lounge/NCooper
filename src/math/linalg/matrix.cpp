/*
    Copyright IdeaLounge.io 2018
*/

#include "math/linalg/matrix.hpp"

namespace ncooper {
namespace math {
namespace linalg {
    template <class MatrixType>
    Matrix<MatrixType>::Matrix(): rows(0), cols(0) {
        this->data = std::vector<Vector<MatrixType>>();
    }

    template <class MatrixType>
    Matrix<MatrixType>::Matrix(int rows, int cols): rows(rows), cols(cols) {
        this->data.reserve(this->rows);
        for (int i = 0; i < this->rows; i++) {
            Vector<MatrixType> vector(this->cols, 0);
            vector.transpose();
            this->data.push_back(vector);
        }
    }

    template <class MatrixType>
    Matrix<MatrixType>::Matrix(int rows, int cols, MatrixType initVal): rows(rows), cols(cols) {
        this->data.reserve(this->rows);
        for (int i = 0; i < this->rows; i++) {
            Vector<MatrixType> vector(this->cols, initVal);
            vector.transpose();
            this->data.push_back(vector);
        }
    }

    template <class MatrixType>
    Matrix<MatrixType>::Matrix(const Matrix<MatrixType>& matrix): rows(matrix.rows), cols(matrix.cols) {
        this->data.reserve(this->rows);
        this->data = matrix.data;
    }

    template <class MatrixType>
    Matrix<MatrixType>::~Matrix() {

    }

    template <class MatrixType>
    void Matrix<MatrixType>::push_back(Vector<MatrixType>& vector) {
        if (this->cols != vector.getSize() && this->cols != 0) {  //checking if the dimensions match
            throw "Cannot add vector to matrix. Vectors dimensions dont match with Matrix dimensions";
        } else {
            vector.transpose(); // transposing vector internally
            this->data.push_back(vector);
            this->rows++;
        }
    }

    template <class MatrixType>
    Matrix<MatrixType> Matrix<MatrixType>::transpose() {
        std::cout << "transposing" << std::endl;
        Matrix<MatrixType> finalMatrix(this->getCols(), this->getRows(), 0);
        for (int i = 0; i < this->getCols(); i++) {
            for (int j = 0; j < this->getRows(); j++) {
                finalMatrix(i, j) = this->data[j][i];
            }
        }
        return finalMatrix;
    }

    template <class MatrixType>
    Matrix<MatrixType> Matrix<MatrixType>::hadamardProduct(const Matrix<MatrixType>& matrix) {
        assert(this->rows == matrix.getRows() && this->cols == matrix.getCols());
        Matrix<MatrixType> finalMatrix(this->rows, this->cols);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                finalMatrix(i, j) = this->operator()(i, j) * matrix(i, j);
            }
        }
        return finalMatrix;
    }

    template <class MatrixType>
    Matrix<MatrixType> Matrix<MatrixType>::kroneckerProduct(const Matrix<MatrixType>& matrix) {
        Matrix<MatrixType> finalMatrix(matrix.getRows() * this->rows,
            matrix.getCols() * this->cols);
        int finalMatrixRow = 0;
        int finalMatrixCol = 0;
        for (int i = 0; i < this->getRows(); i++) {
            for (int j = 0; j < this->getCols(); j++) {
                Matrix<MatrixType> tempMatrix = this->operator()(i, j) * matrix;
                std::cout << "hello" << std::endl;
                std::cout << finalMatrix << std::endl;

                for (int k = 0; k < tempMatrix.getRows(); k++) {
                    for (int l = 0; l < tempMatrix.getCols(); l++) {
                        finalMatrixCol = j * matrix.getCols() + l;
                        finalMatrixRow = i * matrix.getRows() + k;
                        finalMatrix(finalMatrixRow, finalMatrixCol) = tempMatrix(k, l);
                    }
                }
            }
        }
        return finalMatrix;
    }

    template <class MatrixType>
    Matrix<MatrixType> Matrix<MatrixType>::concatRight(const Matrix<MatrixType>& matrix) {
        assert(this->rows == matrix.getRows());
        Matrix<MatrixType> finalMatrix(this->rows, this->cols + matrix.getCols());
        int matrixRow = 0;
        int matrixCol = 0;
        for (int i = 0; i < finalMatrix.getRows(); i++) {
            for (int j = 0; j < finalMatrix.getCols(); j++) {
                if (j < this->cols) {
                    finalMatrix(i, j) = this->operator()(i, j);
                } else {
                    finalMatrix(i, j) = matrix(i, j - this->cols);
                }
            }
        }
        return finalMatrix;
    }

    template <class MatrixType>
    int Matrix<MatrixType>::getRows() const {
        return this->rows;
    }

    template <class MatrixType>
    int Matrix<MatrixType>::getCols() const {
        return this->cols;
    }

    template class Matrix<int>;
    template class Matrix<float>;
}
}
}
