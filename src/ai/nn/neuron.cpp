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
}

template <class NeuronType>
void Neuron<NeuronType>::forwardProp(const ncooper::math::linalg::Vector<NeuronType> &inputVector) {
    std::cout << "WEIGHTS VECTOR: " << *this->weightsVector << std::endl;
    std::cout << "BIAS: " << *this->bias << std::endl;

    NeuronType preActivation = *this->weightsVector * inputVector;

    std::cout << "preactivation: " << preActivation << std::endl;
    *this->output = 1 / (1 + exp(-preActivation));
}

template <class NeuronType>
const NeuronType& Neuron<NeuronType>::getOutput() {
    return *this->output;
}


template class Neuron<int>;
template class Neuron<float>;
}
}
}
