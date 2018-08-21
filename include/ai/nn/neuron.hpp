/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_NEURON_HPP_
#define NCOOPER_NEURON_HPP_

#include <math.h>
#include <iostream>
#include "math/linalg/vector.hpp"


namespace ncooper {
namespace ai {
namespace nn {
template <class NeuronType>
class Neuron {
 public:
    Neuron(int inputVectorSize);
    Neuron(ncooper::math::linalg::Vector<NeuronType> &weightsVector, NeuronType &bias, NeuronType &output);
    ~Neuron();

    void forwardProp(const ncooper::math::linalg::Vector<NeuronType> &inputVector);

    const ncooper::math::linalg::Vector<NeuronType>& getWeightsVector();
    const NeuronType& getBias();
    const NeuronType& getOutput();
 private:
    ncooper::math::linalg::Vector<NeuronType> *weightsVector;  // arbitrary size
    NeuronType *bias;
    NeuronType *output;
    bool soloNeuron;
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_NEURON_HPP_
