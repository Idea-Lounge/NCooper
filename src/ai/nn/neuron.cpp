/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/neuron.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
Neuron<DataType>::Neuron(int inputVectorSize) {
    this->soloNeuron = true;
    this->weightsVector = new ncooper::math::linalg::Vector<DataType>(inputVectorSize, 0);
    this->bias = new DataType(0);
    this->output = new DataType(0);
}

template <class DataType>
Neuron<DataType>::Neuron(ncooper::math::linalg::Vector<DataType> &weightsVector, DataType &bias, DataType &output) {
    this->weightsVector = &weightsVector;
    this->bias = &bias;
    this->output = &output;
    this->soloNeuron = false;
}

template <class DataType>
Neuron<DataType>::~Neuron() {
    if (this->soloNeuron) {
        delete this->weightsVector;
        delete this->bias;
        delete this->output;
    }
}

template <class DataType>
void Neuron<DataType>::forwardProp(const ncooper::math::linalg::Vector<DataType> &inputVector) {
    DataType preActivation = *this->weightsVector * inputVector + *this->bias;
    *this->output = 1 / (1 + exp(-preActivation));
}

template <class DataType>
const ncooper::math::linalg::Vector<DataType>& Neuron<DataType>::getWeightsVector() {
    return *this->weightsVector;
}

template <class DataType>
const DataType& Neuron<DataType>::getBias() {
    return *this->bias;
}

template <class DataType>
const DataType& Neuron<DataType>::getOutput() {
    return *this->output;
}


template class Neuron<int>;
template class Neuron<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
