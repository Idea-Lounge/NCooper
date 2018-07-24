/*
    Copyright IdeaLounge.io 2018
*/

#include "math/linalg/vector.hpp"

namespace ncooper {
namespace math {
namespace linalg {
    template <class VectorType>
    Vector<VectorType>::Vector(): size(0) {
        this->data = std::vector<VectorType>();
    }

    template <class VectorType>
    Vector<VectorType>::Vector(int size): size(size) {
        this->data.reserve(size);
    }

    template <class VectorType>
    Vector<VectorType>::Vector(int size, VectorType initVal): size(size) {
        this->data.reserve(size);
        this->data = std::vector<VectorType>(this->size, initVal);
    }

    template <class VectorType>
    Vector<VectorType>::Vector(const Vector<VectorType>& vector): size(vector.data.size()) {
        this->data.reserve(this->size);
        this->data = vector.data;  // COMBAK: if this does not work, do a manual deep copy.
    }

    template <class VectorType>
    Vector<VectorType>::Vector(const std::vector<VectorType>& vector): size(vector.size()) {
        this->data.reserve(this->size);
        this->data = vector;  // COMBAK: if this does not work, do a manual deep copy.
    }

    template <class VectorType>
    Vector<VectorType>::~Vector() {
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

    template class Vector<int>;
    template class Vector<float>;
}
}
}
