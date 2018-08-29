/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_VECTOR_HPP_
#define NCOOPER_VECTOR_HPP_

#include <assert.h>
#include <iostream>
#include <vector>

namespace ncooper {
namespace math {
namespace linalg {
template <class DataType>
class Vector {
 public:
    Vector();
    Vector(int size, DataType initVal = 0);
    Vector(const Vector<DataType>& vector);
    Vector(const std::vector<DataType>& vector);
    ~Vector();

    void clear();
    void push_back(DataType element);
    void transpose();

    int getSize() const;
    bool isTransposed() const;

    DataType& operator[](int index);
    const DataType& operator[](int index) const;

    Vector<DataType>& operator=(const Vector<DataType>& vector);

 protected:
    std::vector<DataType> data;
    int size;
    bool transposed = false;
};

template <typename DataType>
extern void SVMult(const DataType &scalar,
                   const Vector<DataType> &vector,
                   Vector<DataType> &result);

template <typename DataType>
extern void VVMult(const Vector<DataType> &vector1,
                   const Vector<DataType> &vector2,
                   DataType &result);

template <typename DataType>
extern void VVAdd(const Vector<DataType> &vector1,
                  const Vector<DataType> &vector2,
                  Vector<DataType> &result);

template <typename DataType>
extern Vector<DataType> operator*(const DataType &scalar, const Vector<DataType> &vector);
template <typename DataType>
extern Vector<DataType> operator*(const Vector<DataType> &vector, DataType &scalar);
template <typename DataType>
extern DataType operator*(const Vector<DataType> &vector1, const Vector<DataType> &vector2);
template <typename DataType>
extern Vector<DataType> operator+(const Vector<DataType> &vector1, const Vector<DataType> &vector2);
template <typename DataType>
extern std::ostream& operator<<(std::ostream &os, const Vector<DataType> &vector);

}  // namespace linalg
}  // namespace math
}  // namespace ncooper

#endif  // NCOOPER_VECTOR_HPP_
