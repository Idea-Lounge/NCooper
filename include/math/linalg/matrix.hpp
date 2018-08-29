/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_MATRIX_HPP_
#define NCOOPER_MATRIX_HPP_

#include <assert.h>
#include <iostream>
#include <vector>
#include "vector.hpp"

namespace ncooper {
namespace math {
namespace linalg {
template <class DataType>
class Matrix {
 public:
    Matrix();  // initializes empty matrix
    Matrix(int rows, int cols, DataType initVal = 0);
    Matrix(const Matrix<DataType>& matrix);
    ~Matrix();

    void push_back(Vector<DataType>& vector);
    Matrix<DataType> transpose();
    Matrix<DataType> hadamardProduct(const Matrix<DataType>& matrix);
    Matrix<DataType> kroneckerProduct(const Matrix<DataType>& matrix);
    Matrix<DataType> concatRight(const Matrix<DataType>& matrix);

    int getRows() const;
    int getCols() const;

    DataType& operator()(int row, int col);
    const DataType& operator()(int row, int col) const;
    const Vector<DataType>& operator[](int row) const;
    Vector<DataType>& operator[](int row);
    Matrix<DataType>& operator=(const Matrix<DataType>& matrix);

 protected:
    std::vector<Vector<DataType> > data;
    int rows = 0;
    int cols = 0;
};

template <typename DataType>
extern void MMAdd(const Matrix<DataType> &matrix1,
                  const Matrix<DataType> &matrix2,
                  Matrix<DataType> &result);
template <typename DataType>
extern void MSMult(const Matrix<DataType> &matrix,
                   const DataType &scalar,
                   Matrix<DataType> &result);
template <typename DataType>
extern void MVMult(const Matrix<DataType> &matrix,
                   const Vector<DataType> &vector,
                   Vector<DataType> &result);
template <typename DataType>
extern void MMMult(const Matrix<DataType> &matrix1,
                   const Matrix<DataType> &matrix2,
                   Matrix<DataType> &result);

template <typename DataType>
extern Matrix<DataType> operator+(const Matrix<DataType> &matrix1,
                                    const Matrix<DataType> &matrix2);
template <typename DataType>
extern Matrix<DataType> operator*(const DataType &scalar,
                                    const Matrix<DataType>& matrix);
template <typename DataType>
extern Matrix<DataType> operator*(const Matrix<DataType>& matrix,
                                    const DataType &scalar);
template <typename DataType>
extern Vector<DataType> operator*(const Matrix<DataType>& matrix,
                                    const Vector<DataType>& vector);
template <typename DataType>
extern Matrix<DataType> operator*(const Matrix<DataType>& matrix1,
                                    const Matrix<DataType>& matrix2);
template <typename DataType>
extern std::ostream& operator<<(std::ostream& os,
                                const Matrix<DataType>& matrix);

}  // namespace linalg
}  // namespace math
}  // namespace ncooper

#endif  // NCOOPER_MATRIX_HPP_
