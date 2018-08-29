/*
    Copyright IdeaLounge.io 2018
 */

#include "math/linalg/matrix.hpp"

namespace ncooper {
namespace math {
namespace linalg {

template <class DataType>
Matrix<DataType>::Matrix() : rows(0), cols(0) {
    this->data = std::vector<Vector<DataType> >();
}

template <class DataType>
Matrix<DataType>::Matrix(int rows, int cols, DataType initVal) : rows(rows), cols(cols) {
    this->data.reserve(this->rows);
    for (int i = 0; i < this->rows; i++) {
        Vector<DataType> vector(this->cols, initVal);
        vector.transpose();
        this->data.push_back(vector);
    }
}

template <class DataType>
Matrix<DataType>::Matrix(const Matrix<DataType>& matrix) : rows(matrix.rows), cols(matrix.cols) {
    this->data.reserve(this->rows);
    this->data = matrix.data;
}

template <class DataType>
Matrix<DataType>::~Matrix() {
}

template <class DataType>
void Matrix<DataType>::push_back(Vector<DataType>& vector) {
    assert(this->cols == vector.getSize());
    vector.transpose();     // transposing vector internally
    this->data.push_back(vector);
}

template <class DataType>
Matrix<DataType> Matrix<DataType>::transpose() {
    Matrix<DataType> finalMatrix(this->getCols(), this->getRows(), 0);
    for (int i = 0; i < this->getCols(); i++) {
        for (int j = 0; j < this->getRows(); j++) {
            finalMatrix(i, j) = this->data[j][i];
        }
    }
    return finalMatrix;
}

template <class DataType>
Matrix<DataType> Matrix<DataType>::hadamardProduct(const Matrix<DataType>& matrix) {
    assert(this->rows == matrix.getRows() && this->cols == matrix.getCols());
    Matrix<DataType> finalMatrix(this->rows, this->cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            finalMatrix(i, j) = this->operator()(i, j) * matrix(i, j);
        }
    }
    return finalMatrix;
}

template <class DataType>
Matrix<DataType> Matrix<DataType>::kroneckerProduct(const Matrix<DataType>& matrix) {
    Matrix<DataType> finalMatrix(matrix.getRows() * this->rows,
                                   matrix.getCols() * this->cols);
    int finalMatrixRow = 0;
    int finalMatrixCol = 0;
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            Matrix<DataType> tempMatrix = this->operator()(i, j) * matrix;

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

template <class DataType>
Matrix<DataType> Matrix<DataType>::concatRight(const Matrix<DataType>& matrix) {
    assert(this->rows == matrix.getRows());
    Matrix<DataType> finalMatrix(this->rows, this->cols + matrix.getCols());
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

template <class DataType>
int Matrix<DataType>::getRows() const {
    return this->rows;
}

template <class DataType>
int Matrix<DataType>::getCols() const {
    return this->cols;
}

template <class DataType>
DataType& Matrix<DataType>::operator()(int row, int col) {
    return this->data[row][col];
}

template <class DataType>
const DataType& Matrix<DataType>::operator()(int row, int col) const {
    return this->data[row][col];
}

template <class DataType>
const Vector<DataType>& Matrix<DataType>::operator[](int row) const {
    return this->data[row];
}

template <class DataType>
Vector<DataType>& Matrix<DataType>::operator[](int row) {
    return this->data[row];
}

template <class DataType>
Matrix<DataType>& Matrix<DataType>::operator=(const Matrix<DataType>& matrix) {
    this->rows = matrix.rows;
    this->cols = matrix.cols;
    this->data.reserve(matrix.rows);
    this->data = matrix.data;
    return *this;
}

template <typename DataType>
void MMAdd(const Matrix<DataType> &matrix1,
           const Matrix<DataType> &matrix2,
           Matrix<DataType> &result) {
    assert((matrix1.getCols() == matrix2.getCols() && matrix2.getCols() == result.getCols())
           && (matrix1.getRows() == matrix2.getRows() == result.getRows()));
    for (int i = 0; i < matrix1.getRows(); i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
}

template <typename DataType>
void MSMult(const Matrix<DataType> &matrix,
            const DataType &scalar,
            Matrix<DataType> &result) {
    assert((matrix.getCols() == result.getCols()) && matrix.getRows() == result.getRows());
    for (int i = 0; i < matrix.getRows(); i++) {
        result[i] = scalar * matrix[i];
    }
}

template <typename DataType>
void MVMult(const Matrix<DataType> &matrix,
            const Vector<DataType> &vector,
            Vector<DataType> &result) {
    assert(matrix.getCols() == vector.getSize() && matrix.getRows() == result.getSize());
    for (int i = 0; i < matrix.getRows(); i++) {
        result[i] = matrix[i] * vector;
    }
}

template <typename DataType>
void MMMult(const Matrix<DataType> &matrix1,
            const Matrix<DataType> &matrix2,
            Matrix<DataType> &result) {
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

template <typename DataType>
Matrix<DataType> operator+(const Matrix<DataType> &matrix1, const Matrix<DataType> &matrix2) {
    assert(matrix1.getCols() == matrix2.getCols() && matrix1.getRows() == matrix2.getRows());
    Matrix<DataType> finalMatrix(matrix1.getRows(), matrix1.getCols());
    MMAdd(matrix1, matrix2, finalMatrix);
    return finalMatrix;
}

template <typename DataType>
Matrix<DataType> operator*(const DataType &scalar, const Matrix<DataType> &matrix) {
    Matrix<DataType> finalMatrix(matrix.getRows(), matrix.getCols(), 0);
    MSMult(matrix, scalar, finalMatrix);
    return finalMatrix;
}

template <typename DataType>
Matrix<DataType> operator*(const Matrix<DataType>& matrix, const DataType &scalar) {
    Matrix<DataType> finalMatrix(matrix.getRows(), matrix.getCols(), 0);
    MSMult(matrix, scalar, finalMatrix);
    return finalMatrix;
}

template <typename DataType>
Vector<DataType> operator*(const Matrix<DataType> &matrix, const Vector<DataType> &vector) {
    assert(matrix.getCols() == vector.getSize());
    Vector<DataType> finalVector(matrix.getRows(), 0);
    MVMult(matrix, vector, finalVector);
    return finalVector;
}

template <typename DataType>
Matrix<DataType> operator*(const Matrix<DataType>& matrix1, const Matrix<DataType>& matrix2) {
    assert(matrix1.getCols() == matrix2.getRows());
    Matrix<DataType> finalMatrix(matrix1.getRows(), matrix2.getCols(), 0);
    MMMult(matrix1, matrix2, finalMatrix);
    return finalMatrix;
}

template <typename DataType>
std::ostream& operator<<(std::ostream& os, const Matrix<DataType>& matrix) {
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

}  // namespace linalg
}  // namespace math
}  // namespace ncooper
