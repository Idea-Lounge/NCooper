/*
    Copyright IdeaLounge.io 2018
 */

#include "math/linalg/matrix.hpp"

namespace ncooper {
namespace math {
namespace linalg {

template <class MatrixType>
Matrix<MatrixType>::Matrix() : rows(0), cols(0) {
    this->data = std::vector<Vector<MatrixType>>();
}

template <class MatrixType>
Matrix<MatrixType>::Matrix(int rows, int cols, MatrixType initVal) : rows(rows), cols(cols) {
    this->data.reserve(this->rows);
    for (int i = 0; i < this->rows; i++) {
        Vector<MatrixType> vector(this->cols, initVal);
        vector.transpose();
        this->data.push_back(vector);
    }
}

template <class MatrixType>
Matrix<MatrixType>::Matrix(const Matrix<MatrixType>& matrix) : rows(matrix.rows), cols(matrix.cols) {
    this->data.reserve(this->rows);
    this->data = matrix.data;
}

template <class MatrixType>
Matrix<MatrixType>::~Matrix() {}

template <class MatrixType>
void Matrix<MatrixType>::push_back(Vector<MatrixType>& vector) {
    assert(this->cols == vector.getSize());
    vector.transpose();     // transposing vector internally
    this->data.push_back(vector);
}

template <class MatrixType>
Matrix<MatrixType> Matrix<MatrixType>::transpose() {
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

template <class MatrixType>
MatrixType& Matrix<MatrixType>::operator()(int row, int col) {
    return this->data[row][col];
}

template <class MatrixType>
const MatrixType& Matrix<MatrixType>::operator()(int row, int col) const {
    return this->data[row][col];
}

template <class MatrixType>
const Vector<MatrixType>& Matrix<MatrixType>::operator[](int row) const {
    return this->data[row];
}

template <class MatrixType>
Vector<MatrixType>& Matrix<MatrixType>::operator[](int row) {
    return this->data[row];
}

template <class MatrixType>
Matrix<MatrixType>& Matrix<MatrixType>::operator=(const Matrix<MatrixType>& matrix) {
    this->rows = matrix.rows;
    this->cols = matrix.cols;
    this->data.reserve(matrix.rows);
    this->data = matrix.data;
    return *this;
}

template <typename MatrixType>
void MMAdd(const Matrix<MatrixType> &matrix1,
    const Matrix<MatrixType> &matrix2,
    Matrix<MatrixType> &result) {
    assert((matrix1.getCols() == matrix2.getCols() && matrix2.getCols() == result.getCols())
        && (matrix1.getRows() == matrix2.getRows() == result.getRows()));
    for (int i = 0; i < matrix1.getRows(); i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
}

template <typename MatrixType>
void MSMult(const Matrix<MatrixType> &matrix,
    const MatrixType &scalar,
    Matrix<MatrixType> &result) {
    assert((matrix.getCols() == result.getCols()) && matrix.getRows() == result.getRows());
    for (int i = 0; i < matrix.getRows(); i++) {
        result[i] = scalar * matrix[i];
    }
}

template <typename MatrixType>
void MVMult(const Matrix<MatrixType> &matrix,
    const Vector<MatrixType> &vector,
    Vector<MatrixType> &result) {
    assert(matrix.getCols() == vector.getSize() && matrix.getRows() == result.getSize());
    for (int i = 0; i < matrix.getRows(); i++) {
        result[i] = matrix[i] * vector;
    }
}

template <typename MatrixType>
void MMMult(const Matrix<MatrixType> &matrix1,
    const Matrix<MatrixType> &matrix2,
    Matrix<MatrixType> &result) {
    assert((matrix1.getCols() == matrix2.getRows())
        && (result.getRows() == matrix1.getRows())
        && (result.getCols() == matrix2.getCols()));
    for (int i = 0; i < matrix1.getRows(); i++) {
        for (int j = 0; j < matrix2.getCols(); j++) {
            for (int k = 0; k < matrix1.getCols(); k++) {
                result(i, j) += matrix1(i, k) * matrix2(k, j);
            }
        }
    }
}

template <typename MatrixType>
Matrix<MatrixType> operator+(const Matrix<MatrixType> &matrix1, const Matrix<MatrixType> &matrix2) {
    assert(matrix1.getCols() == matrix2.getCols() && matrix1.getRows() == matrix2.getRows());
    Matrix<MatrixType> finalMatrix(matrix1.getRows(), matrix1.getCols());
    MMAdd(matrix1, matrix2, finalMatrix);
    return finalMatrix;
}

template <typename MatrixType>
Matrix<MatrixType> operator*(const MatrixType &scalar, const Matrix<MatrixType> &matrix) {
    Matrix<MatrixType> finalMatrix(matrix.getRows(), matrix.getCols(), 0);
    MSMult(matrix, scalar, finalMatrix);
    return finalMatrix;
}

template <typename MatrixType>
Matrix<MatrixType> operator*(const Matrix<MatrixType>& matrix, const MatrixType &scalar) {
    Matrix<MatrixType> finalMatrix(matrix.getRows(), matrix.getCols(), 0);
    MSMult(matrix, scalar, finalMatrix);
    return finalMatrix;
}

template <typename MatrixType>
Vector<MatrixType> operator*(const Matrix<MatrixType> &matrix, const Vector<MatrixType> &vector) {
    assert(matrix.getCols() == vector.getSize());
    Vector<MatrixType> finalVector(matrix.getRows(), 0);
    MVMult(matrix, vector, finalVector);
    return finalVector;
}

template <typename MatrixType>
Matrix<MatrixType> operator*(const Matrix<MatrixType>& matrix1, const Matrix<MatrixType>& matrix2) {
    assert(matrix1.getCols() == matrix2.getRows());
    Matrix<MatrixType> finalMatrix(matrix1.getRows(), matrix2.getCols(), 0);
    MMMult(matrix1, matrix2, finalMatrix);
    return finalMatrix;
}

template <typename MatrixType>
std::ostream& operator<<(std::ostream& os, const Matrix<MatrixType>& matrix) {
    for (int i = 0; i < matrix.getRows(); i++) {
        os << matrix[i] << '\n';
    }
    return os;
}

template class Matrix<int>;
template class Matrix<float>;

template void MMAdd<int>(const Matrix<int>&, const Matrix<int>&, Matrix<int>&);
template void MMAdd<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void MSMult<int>(const Matrix<int>&, const int&, Matrix<int>&);
template void MSMult<float>(const Matrix<float>&, const float&, Matrix<float>&);
template void MVMult<int>(const Matrix<int> &matrix, const Vector<int>&, Vector<int>&);
template void MVMult<float>(const Matrix<float> &matrix, const Vector<float>&, Vector<float>&);
template void MMMult<int>(const Matrix<int>&, const Matrix<int>&, Matrix<int>&);
template void MMMult<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);

template Matrix<int> operator+<int>(const Matrix<int>&, const Matrix<int>&);
template Matrix<float> operator+<float>(const Matrix<float>&, const Matrix<float>&);
template Matrix<int> operator*<int>(const int&, const Matrix<int>&);
template Matrix<float> operator*<float>(const float&, const Matrix<float>&);
template Matrix<int> operator*<int>(const Matrix<int>&, const int&);
template Matrix<float> operator*<float>(const Matrix<float>&, const float&);
template Vector<int> operator*<int>(const Matrix<int>&, const Vector<int>&);
template Vector<float> operator*<float>(const Matrix<float>&, const Vector<float>&);
template Matrix<int> operator*<int>(const Matrix<int>&, const Matrix<int>&);
template Matrix<float> operator*<float>(const Matrix<float>&, const Matrix<float>&);
template std::ostream& operator<<<int>(std::ostream&, const Matrix<int>&);
template std::ostream& operator<<<float>(std::ostream&, const Matrix<float>&);

}
}
}
