/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_VECTOR_HPP_
#define NCOOPER_VECTOR_HPP_

#include <vector>
#include <iostream>
#include <assert.h>

namespace ncooper {
namespace math {
namespace linalg {
template <class VectorType>
class Vector {
 public:
    Vector();
    Vector(int size, VectorType initVal = 0);
    Vector(const Vector<VectorType>& vector);
    Vector(const std::vector<VectorType>& vector);
    ~Vector();

    void clear();
    void push_back(VectorType element);
    void transpose();

    int getSize() const;
    bool isTransposed() const;

    VectorType& operator[](int index);
    const VectorType& operator[](int index) const;

    Vector<VectorType>& operator=(const Vector<VectorType>& vector);

 protected:
    std::vector<VectorType> data;
    int size;
    bool transposed = false;
};

template <typename VectorType>
extern void SVMult(const VectorType &scalar,
    const Vector<VectorType> &vector,
    Vector<VectorType> &result);

template <typename VectorType>
extern void VVMult(const Vector<VectorType> &vector1,
    const Vector<VectorType> &vector2,
    VectorType &result);

template <typename VectorType>
extern void VVAdd(const Vector<VectorType> &vector1,
    const Vector<VectorType> &vector2,
    Vector<VectorType> &result);

template <typename VectorType>
extern Vector<VectorType> operator*(const VectorType &scalar, const Vector<VectorType> &vector);
template <typename VectorType>
extern Vector<VectorType> operator*(const Vector<VectorType> &vector, VectorType &scalar);
template <typename VectorType>
extern VectorType operator*(const Vector<VectorType> &vector1, const Vector<VectorType> &vector2);
template <typename VectorType>
extern Vector<VectorType> operator+(const Vector<VectorType> &vector1, const Vector<VectorType> &vector2);
template <typename VectorType>
extern std::ostream& operator<<(std::ostream &os, const Vector<VectorType> &vector);

}
}
}

#endif  // NCOOPER_VECTOR_HPP_
