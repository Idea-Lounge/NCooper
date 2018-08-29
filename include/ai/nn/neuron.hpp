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
template <class DataType>
class Neuron {
 public:
    Neuron(int inputVectorSize);
    Neuron(ncooper::math::linalg::Vector<DataType> &weightsVector, DataType &bias, DataType &output);
    ~Neuron();

    void forwardProp(const ncooper::math::linalg::Vector<DataType> &inputVector);

    const ncooper::math::linalg::Vector<DataType>& getWeightsVector();
    const DataType& getBias();
    const DataType& getOutput();
 private:
    ncooper::math::linalg::Vector<DataType> *weightsVector;  // arbitrary size
    DataType *bias;
    DataType *output;
    bool soloNeuron;
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_NEURON_HPP_
