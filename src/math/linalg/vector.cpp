/*
    Copyright IdeaLounge.io 2018
 */

#include "math/linalg/vector.hpp"

namespace ncooper {
namespace math {
namespace linalg {
template <class VectorType>
Vector<VectorType>::Vector() : size(0) {
    this->data = std::vector<VectorType>();
}

template <class VectorType>
Vector<VectorType>::Vector(int size, VectorType initVal) : size(size) {
    this->data.reserve(size);
    this->data = std::vector<VectorType>(this->size, initVal);
}

template <class VectorType>
Vector<VectorType>::Vector(const Vector<VectorType>& vector) : size(vector.data.size()) {
    this->data.reserve(this->size);
    this->data = vector.data;      // COMBAK: if this does not work, do a manual deep copy.
    this->transposed = vector.transposed;
}

template <class VectorType>
Vector<VectorType>::Vector(const std::vector<VectorType>& vector) : size(vector.size()) {
    this->data.reserve(this->size);
    this->data = vector;      // COMBAK: if this does not work, do a manual deep copy.
}

template <class VectorType>
Vector<VectorType>::~Vector() {}

template <class VectorType>
void Vector<VectorType>::clear() {
    this->data.clear();
    this->size = 0;
}

template <class VectorType>
void Vector<VectorType>::push_back(VectorType element) {
    this->data.push_back(element);
    this->size++;
}

template <class VectorType>
int Vector<VectorType>::getSize() const {
    return this->size;
}

template <class VectorType>
void Vector<VectorType>::transpose() {
    this->transposed = !this->transposed;
}

template <class VectorType>
bool Vector<VectorType>::isTransposed() const {
    return this->transposed;
}

template <class VectorType>
VectorType& Vector<VectorType>::operator[](int index) {
    return this->data[index];
}

template <class VectorType>
const VectorType& Vector<VectorType>::operator[](int index) const {
    return this->data[index];
}

template <class VectorType>
Vector<VectorType>& Vector<VectorType>::operator=(const Vector<VectorType>& vector) {
    this->data = vector.data;
    this->transposed = vector.transposed;
    this->size = vector.getSize();
    return *this;
}

template <typename VectorType>
void SVMult(const VectorType &scalar,
    const Vector<VectorType> &vector,
    Vector<VectorType> &result) {
    assert(vector.getSize() == result.getSize());
    for (int i = 0; i < vector.getSize(); i++) {
        result[i] = vector[i] * scalar;
    }
}

template <typename VectorType>
void VVMult(const Vector<VectorType> &vector1,
    const Vector<VectorType> &vector2,
    VectorType &result) {
    assert(vector1.isTransposed() && vector1.getSize() == vector2.getSize());
    result = 0;
    for (int i = 0; i < vector1.getSize(); i++) {
        result += vector1[i] * vector2[i];
    }
}

template <typename VectorType>
void VVAdd(const Vector<VectorType> &vector1,
    const Vector<VectorType> &vector2,
    Vector<VectorType> &result) {
    assert(vector1.getSize() == vector2.getSize() && vector2.getSize() == result.getSize());
    for (int i = 0; i < vector1.getSize(); i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

template <typename VectorType>
Vector<VectorType> operator*(const VectorType &scalar, const Vector<VectorType> &vector) {
    Vector<VectorType> result(vector.getSize());
    SVMult(scalar, vector, result);
    return result;
}

template <typename VectorType>
Vector<VectorType> operator*(const Vector<VectorType> &vector, const VectorType &scalar) {
    Vector<VectorType> result(vector.getSize());
    SVMult(scalar, vector, result);
    return result;
}

template <typename VectorType>
VectorType operator*(const Vector<VectorType> &vector1, const Vector<VectorType> &vector2) {
    assert(vector1.isTransposed() && vector1.getSize() == vector2.getSize());
    VectorType result;
    VVMult(vector1, vector2, result);
    return result;
}

template <typename VectorType>
Vector<VectorType> operator+(const Vector<VectorType> &vector1, const Vector<VectorType> &vector2) {
    assert((vector1.getSize() == vector2.getSize())
        && (vector1.isTransposed() == vector2.isTransposed()));
    Vector<VectorType> result(vector1.getSize());
    VVAdd(vector1, vector2, result);
    return result;
}

template <typename VectorType>
std::ostream& operator<<(std::ostream& os, const Vector<VectorType> &vector) {
    for (int i = 0; i < vector.getSize(); i++) {
        os << vector[i] << " ";
    }
    return os;
}

template class Vector<int>;
template class Vector<float>;
template void SVMult<int>(const int&, const Vector<int>&, Vector<int>&);
template void SVMult<float>(const float&, const Vector<float>&, Vector<float>&);
template void VVMult<int>(const Vector<int>&, const Vector<int>&, int&);
template void VVMult<float>(const Vector<float>&, const Vector<float>&, float&);
template void VVAdd<int>(const Vector<int>&, const Vector<int>&, Vector<int>&);
template void VVAdd<float>(const Vector<float>&, const Vector<float>&, Vector<float>&);
template Vector<int> operator*<int>(const int&, const Vector<int>&);
template Vector<float> operator*<float>(const float&, const Vector<float>&);
template Vector<int> operator*<int>(const Vector<int>&, const int&);
template Vector<float> operator*<float>(const Vector<float>&, const float&);
template int operator*<int>(const Vector<int>&, const Vector<int>&);
template float operator*<float>(const Vector<float>&, const Vector<float>&);
template Vector<int> operator+<int>(const Vector<int>&, const Vector<int>&);
template Vector<float> operator+<float>(const Vector<float>&, const Vector<float>&);
template std::ostream& operator<<<int>(std::ostream&, const Vector<int>&);
template std::ostream& operator<<<float>(std::ostream&, const Vector<float>&);

}
}
}
