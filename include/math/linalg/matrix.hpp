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
    Matrix();                 // initializes empty matrix
    Matrix(int rows, int cols, MatrixType initVal = 0);
    Matrix(const Matrix<MatrixType>& matrix);
    ~Matrix();

    void push_back(Vector<MatrixType>& vector);
    Matrix<MatrixType> transpose();
    Matrix<MatrixType> hadamardProduct(const Matrix<MatrixType>& matrix);
    Matrix<MatrixType> kroneckerProduct(const Matrix<MatrixType>& matrix);
    Matrix<MatrixType> concatRight(const Matrix<MatrixType>& matrix);

    int getRows() const;
    int getCols() const;

    MatrixType& operator()(int row, int col);
    const MatrixType& operator()(int row, int col) const;
    const Vector<MatrixType>& operator[](int row) const;
    Vector<MatrixType>& operator[](int row);

 protected:
    std::vector<Vector<MatrixType>> data;
    int rows = 0;
    int cols = 0;
};

template <typename MatrixType>
extern void MMAdd(const Matrix<MatrixType> &matrix1,
    const Matrix<MatrixType> &matrix2,
    Matrix<MatrixType> &result);
template <typename MatrixType>
extern void MSMult(const Matrix<MatrixType> &matrix,
    const MatrixType &scalar,
    Matrix<MatrixType> &result);
template <typename MatrixType>
extern void MVMult(const Matrix<MatrixType> &matrix,
    const Vector<MatrixType> &vector,
    Vector<MatrixType> &result);
template <typename MatrixType>
extern void MMMult(const Matrix<MatrixType> &matrix1,
    const Matrix<MatrixType> &matrix2,
    Matrix<MatrixType> &result);

template <typename MatrixType>
extern Matrix<MatrixType> operator+(const Matrix<MatrixType> &matrix1,
    const Matrix<MatrixType> &matrix2);
template <typename MatrixType>
extern Matrix<MatrixType> operator*(const MatrixType &scalar,
    const Matrix<MatrixType>& matrix);
template <typename MatrixType>
extern Matrix<MatrixType> operator*(const Matrix<MatrixType>& matrix,
    const MatrixType &scalar);
template <typename MatrixType>
extern Vector<MatrixType> operator*(const Matrix<MatrixType>& matrix,
    const Vector<MatrixType>& vector);
template <typename MatrixType>
extern Matrix<MatrixType> operator*(const Matrix<MatrixType>& matrix1,
    const Matrix<MatrixType>& matrix2);
template <typename MatrixType>
extern std::ostream& operator<<(std::ostream& os,
    const Matrix<MatrixType>& matrix);

}
}
}

#endif  // NCOOPER_MATRIX_HPP_
