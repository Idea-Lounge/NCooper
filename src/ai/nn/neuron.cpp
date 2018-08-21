/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/neuron.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class NeuronType>
Neuron<NeuronType>::Neuron(int inputVectorSize) {
    this->soloNeuron = true;
    this->weightsVector = new ncooper::math::linalg::Vector<NeuronType>(inputVectorSize, 0);
    this->bias = new NeuronType(0);
    this->output = new NeuronType(0);
}

template <class NeuronType>
Neuron<NeuronType>::Neuron(ncooper::math::linalg::Vector<NeuronType> &weightsVector, NeuronType &bias, NeuronType &output) {
    this->weightsVector = &weightsVector;
    this->bias = &bias;
    this->output = &output;
    this->soloNeuron = false;
}

template <class NeuronType>
Neuron<NeuronType>::~Neuron() {
    if (this->soloNeuron) {
        delete this->weightsVector;
        delete this->bias;
        delete this->output;
    }
}

template <class NeuronType>
void Neuron<NeuronType>::forwardProp(const ncooper::math::linalg::Vector<NeuronType> &inputVector) {
    NeuronType preActivation = *this->weightsVector * inputVector + *this->bias;
    *this->output = 1 / (1 + exp(-preActivation));
}

template <class NeuronType>
const ncooper::math::linalg::Vector<NeuronType>& Neuron<NeuronType>::getWeightsVector() {
    return *this->weightsVector;
}

template <class NeuronType>
const NeuronType& Neuron<NeuronType>::getBias() {
    return *this->bias;
}

template <class NeuronType>
const NeuronType& Neuron<NeuronType>::getOutput() {
    return *this->output;
}


template class Neuron<int>;
template class Neuron<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
