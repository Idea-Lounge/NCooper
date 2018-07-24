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
                    Vector(int size);
                    Vector(int size, VectorType initVal = 0);
                    Vector(const Vector<VectorType>& vector);
                    Vector(const std::vector<VectorType>& vector);
                    ~Vector();

                    void push_back(VectorType element);
                    void transpose();

                    int getSize() const;
                    bool isTransposed() const;
                    // Multiply with scalar
                    friend Vector<VectorType> operator*(VectorType scalar,
                        const Vector<VectorType>& vector) {
                        Vector<VectorType> finalVector;
                        for (auto element = vector.data.begin(); element != vector.data.end(); ++element) {
                            finalVector.push_back(*element * scalar);
                        }
                        return finalVector;
                    }

                    friend Vector<VectorType> operator*(const Vector<VectorType>& vector,
                        VectorType scalar) {
                        Vector<VectorType> finalVector;
                        for (auto element = vector.data.begin(); element != vector.data.end(); ++element) {
                            finalVector.push_back(*element * scalar);
                        }
                        return finalVector;
                    }

                    // Vector Vector Multiplication
                    friend VectorType operator*(const Vector<VectorType>& vector1, const Vector<VectorType>& vector2) {
                        assert(vector1.isTransposed() && vector1.getSize() == vector2.getSize());
                        VectorType dotProduct = 0;
                        for (int i = 0; i < vector1.getSize(); i++) {
                            dotProduct += vector1[i] * vector2[i];
                        }
                        return dotProduct;
                    }

                    // Vector addition
                    friend Vector<VectorType> operator+(const Vector<VectorType>& vector1,
                        const Vector<VectorType>& vector2) {
                        Vector<VectorType> finalVector;
                        if (vector1.getSize() != vector2.getSize()) {
                            throw "Vectors cannot be added due to size mismatch!";
                        }
                        for (int i = 0; i < vector1.getSize(); i++) {
                            finalVector.push_back(vector1[i] + vector2[i]);
                        }
                        return finalVector;
                    }

                    friend std::ostream& operator<<(std::ostream& os, const Vector<VectorType>& vector) {
                		for (int i = 0; i < vector.getSize(); i++){
                			os << vector[i] << " ";
                		}
                	    return os;
                	}

                    VectorType& operator[](int index) {
                        return this->data[index];
                    }

                    Vector<VectorType>& operator=(const Vector<VectorType>& vector) {
                        assert(this->size == vector.getSize());
                        this->data = vector.data;
                        return *this;
                    }

                    const VectorType& operator[](int index) const {
                        return this->data[index];
                    }

                protected:
                    std::vector<VectorType> data;
                    int size;
                    bool transposed = false;
            };
        }
    }
}

#endif  // NCOOPER_VECTOR_HPP_
