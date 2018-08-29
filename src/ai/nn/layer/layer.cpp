/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/layer/layer.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
Layer<DataType>::Layer(int inputVectorSize, int outputVectorSize) :
    inputVectorSize(inputVectorSize), outputVectorSize(outputVectorSize) {
    this->outputVector = ncooper::math::linalg::Vector<DataType>(outputVectorSize);
}

template <class DataType>
Layer<DataType>::~Layer() {

}

template <class DataType>
void Layer<DataType>::forwardProp(const ncooper::math::linalg::Vector<DataType>& inputVector) {};

template <class DataType>
const int Layer<DataType>::getInputVectorSize() {
    return this->inputVectorSize;
}

template <class DataType>
const ncooper::math::linalg::Vector<DataType>& Layer<DataType>::getOutputVector() {
    return this->outputVector;
}

template class Layer<int>;
template class Layer<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
