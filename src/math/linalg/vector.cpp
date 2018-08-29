/*
    Copyright IdeaLounge.io 2018
 */

#include "math/linalg/vector.hpp"

namespace ncooper {
namespace math {
namespace linalg {
template <class DataType>
Vector<DataType>::Vector() : size(0) {
    this->data = std::vector<DataType>();
}

template <class DataType>
Vector<DataType>::Vector(int size, DataType initVal) : size(size) {
    this->data.reserve(size);
    this->data = std::vector<DataType>(this->size, initVal);
}

template <class DataType>
Vector<DataType>::Vector(const Vector<DataType>& vector) : size(vector.data.size()) {
    this->data.reserve(this->size);
    this->data = vector.data;      // COMBAK: if this does not work, do a manual deep copy.
    this->transposed = vector.transposed;
}

template <class DataType>
Vector<DataType>::Vector(const std::vector<DataType>& vector) : size(vector.size()) {
    this->data.reserve(this->size);
    this->data = vector;      // COMBAK: if this does not work, do a manual deep copy.
}

template <class DataType>
Vector<DataType>::~Vector() {
}

template <class DataType>
void Vector<DataType>::clear() {
    this->data.clear();
    this->size = 0;
}

template <class DataType>
void Vector<DataType>::push_back(DataType element) {
    this->data.push_back(element);
    this->size++;
}

template <class DataType>
int Vector<DataType>::getSize() const {
    return this->size;
}

template <class DataType>
void Vector<DataType>::transpose() {
    this->transposed = !this->transposed;
}

template <class DataType>
bool Vector<DataType>::isTransposed() const {
    return this->transposed;
}

template <class DataType>
DataType& Vector<DataType>::operator[](int index) {
    return this->data[index];
}

template <class DataType>
const DataType& Vector<DataType>::operator[](int index) const {
    return this->data[index];
}

template <class DataType>
Vector<DataType>& Vector<DataType>::operator=(const Vector<DataType>& vector) {
    this->data = vector.data;
    this->transposed = vector.transposed;
    this->size = vector.getSize();
    return *this;
}

template <typename DataType>
void SVMult(const DataType &scalar,
            const Vector<DataType> &vector,
            Vector<DataType> &result) {
    assert(vector.getSize() == result.getSize());
    for (int i = 0; i < vector.getSize(); i++) {
        result[i] = vector[i] * scalar;
    }
}

template <typename DataType>
void VVMult(const Vector<DataType> &vector1,
            const Vector<DataType> &vector2,
            DataType &result) {
    assert(vector1.isTransposed() && vector1.getSize() == vector2.getSize());
    result = 0;
    for (int i = 0; i < vector1.getSize(); i++) {
        result += vector1[i] * vector2[i];
    }
}

template <typename DataType>
void VVAdd(const Vector<DataType> &vector1,
           const Vector<DataType> &vector2,
           Vector<DataType> &result) {
    assert(vector1.getSize() == vector2.getSize() && vector2.getSize() == result.getSize());
    for (int i = 0; i < vector1.getSize(); i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

template <typename DataType>
Vector<DataType> operator*(const DataType &scalar, const Vector<DataType> &vector) {
    Vector<DataType> result(vector.getSize());
    SVMult(scalar, vector, result);
    return result;
}

template <typename DataType>
Vector<DataType> operator*(const Vector<DataType> &vector, const DataType &scalar) {
    Vector<DataType> result(vector.getSize());
    SVMult(scalar, vector, result);
    return result;
}

template <typename DataType>
DataType operator*(const Vector<DataType> &vector1, const Vector<DataType> &vector2) {
    assert(vector1.isTransposed() && vector1.getSize() == vector2.getSize());
    DataType result;
    VVMult(vector1, vector2, result);
    return result;
}

template <typename DataType>
Vector<DataType> operator+(const Vector<DataType> &vector1, const Vector<DataType> &vector2) {
    assert((vector1.getSize() == vector2.getSize())
           && (vector1.isTransposed() == vector2.isTransposed()));
    Vector<DataType> result(vector1.getSize());
    VVAdd(vector1, vector2, result);
    return result;
}

template <typename DataType>
std::ostream& operator<<(std::ostream& os, const Vector<DataType> &vector) {
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

}  // namespace linalg
}  // namespace math
}  // namespace ncooper
