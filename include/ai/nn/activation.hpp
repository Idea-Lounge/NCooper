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

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
class Activation {
 public:
    Activation();
    Activation(std::string activationName);
    DataType computeActivation(DataType z);

 private:
    DataType identity(DataType z);
    DataType sigmoid(DataType z);
    DataType tanh(DataType z);
    DataType relu(DataType z);
    DataType leakyRelu(DataType z);
    std::string activationName;
    const std::vector<std::string> activationsList = {"identity", "sigmoid", "tanh", "relu", "leakyRelu"};
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_AI_NN_LAYER_ACTIVATION_LAYER_HPP_
