/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/layer/activationLayer.hpp"

namespace ncooper {
namespace ai {
namespace nn {

template <class DataType>
ActivationLayer<DataType>::ActivationLayer(int inputVectorSize,
                                           int outputVectorSize,
                                           std::string activationType) :
    Layer<DataType>(inputVectorSize, outputVectorSize),
    activationType(activationType) {
    this->outputVector = ncooper::math::linalg::Vector<DataType>(outputVectorSize);
}

template <class DataType>
void ActivationLayer<DataType>::forwardProp(
    const ncooper::math::linalg::Vector<DataType>& inputVector) {
    assert(this->inputVectorSize == inputVector.getSize());
    for (int i = 0; i < this->inputVectorSize; i++) {
        this->outputVector[i] = this->computeActivation(inputVector[i]);
    }
}

// Use enum and replace these if-else-if statements
template <class DataType>
DataType ActivationLayer<DataType>::computeActivation(DataType z) {
    if (this->activationType == "identity") {
        return identity(z);
    } else if (this->activationType == "sigmoid") {
        return sigmoid(z);
    } else if (this->activationType == "tanh") {
        return this->tanh(z);
    } else if (this->activationType == "relu") {
        return this->relu(z);
    } else if (this->activationType == "leakyRelu") {
        return this->leakyRelu(z);
    }
}

template <class DataType>
DataType ActivationLayer<DataType>::identity(DataType z) {
    return (z);
}

template <class DataType>
DataType ActivationLayer<DataType>::sigmoid(DataType z) {
    return (1 / (1 + std::exp(-z)));
}

template <class DataType>
DataType ActivationLayer<DataType>::tanh(DataType z) {
    return ((std::exp(z) - std::exp(-z)) / ((std::exp(z) + std::exp(-z))));
}

template <class DataType>
DataType ActivationLayer<DataType>::relu(DataType z) {
    return ((z >= 0) ? z : 0.0);
}

template <class DataType>
DataType ActivationLayer<DataType>::leakyRelu(DataType z) {
    return ((z >= 0) ? z : (0.01 * z));
}

template class ActivationLayer<int>;
template class ActivationLayer<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
