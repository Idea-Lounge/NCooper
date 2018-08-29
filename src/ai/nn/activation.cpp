/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/activation.hpp"

namespace ncooper {
namespace ai {
namespace nn {

template <class DataType>
Activation<DataType>::Activation() {
    std::cout << "No activation name has been specified." << std::endl;
    this->activationName = "relu";
    std::cout << "Default activation relu chosen" << std::endl;
}

template <class DataType>
Activation<DataType>::Activation(std::string activationName) {
    if (std::find(this->activationsList.begin(), this->activationsList.end(), activationName) != this->activationsList.end()) {
        this->activationName = activationName;
    } else {
        this->activationName = "relu";
        std::cout << "Default activation relu chosen" << std::endl;
    }
}


// Use enum and replace these if-else-if statements
template <class DataType>
DataType Activation<DataType>::computeActivation(DataType z) {
    if (this->activationName == "identity") {
        return identity(z);
    } else if (this->activationName == "sigmoid") {
        return sigmoid(z);
    } else if (this->activationName == "tanh") {
        return this->tanh(z);
    } else if (this->activationName == "relu") {
        return this->relu(z);
    } else if (this->activationName == "leakyRelu") {
        return this->leakyRelu(z);
    }
}

template <class DataType>
DataType Activation<DataType>::identity(DataType z) {
    return (z);
}

template <class DataType>
DataType Activation<DataType>::sigmoid(DataType z) {
    return (1 / (1 + std::exp(-z)));
}

template <class DataType>
DataType Activation<DataType>::tanh(DataType z) {
    return ((std::exp(z) - std::exp(-z)) / ((std::exp(z) + std::exp(-z))));
}

template <class DataType>
DataType Activation<DataType>::relu(DataType z) {
    return ((z >= 0) ? z : 0.0);
}

template <class DataType>
DataType Activation<DataType>::leakyRelu(DataType z) {
    return ((z >= 0) ? z : (0.01 * z));
}

template class Activation<int>;
template class Activation<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
