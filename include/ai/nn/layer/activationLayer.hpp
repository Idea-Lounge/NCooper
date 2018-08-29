/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_AI_NN_LAYER_ACTIVATION_LAYER_HPP_
#define NCOOPER_AI_NN_LAYER_ACTIVATION_LAYER_HPP_

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include "ai/nn/layer/layer.hpp"
#include "math/linalg/vector.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
class ActivationLayer: public Layer<DataType> {
 public:
    ActivationLayer(int inputVectorSize,
                    int outputVectorSize,
                    std::string activationType = "sigmoid");
    ~ActivationLayer();

    void forwardProp(const ncooper::math::linalg::Vector<DataType>& inputVector);
    std::string getActivationType();
    // change to forward Prop

 private:
    DataType computeActivation(DataType z);
    DataType identity(DataType z);
    DataType sigmoid(DataType z);
    DataType tanh(DataType z);
    DataType relu(DataType z);
    DataType leakyRelu(DataType z);
    std::string activationType;
    const std::vector<std::string> activationsList = {"identity", "sigmoid", "tanh", "relu", "leakyRelu"};

    // add activation neuron vector
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_AI_NN_LAYER_ACTIVATION_LAYER_HPP_
