/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_ACTIVATION_HPP_
#define NCOOPER_ACTIVATION_HPP_

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

namespace ncooper {
namespace ai {
namespace nn {
template <class ActivationType>
class Activation {
 public:
    Activation();
    Activation(std::string activationName);
    ActivationType computeActivation(ActivationType z);

 private:
    ActivationType identity(ActivationType z);
    ActivationType sigmoid(ActivationType z);
    ActivationType tanh(ActivationType z);
    ActivationType relu(ActivationType z);
    ActivationType leakyRelu(ActivationType z);
    std::string activationName;
    const std::vector<std::string> activationsList = {"identity", "sigmoid", "tanh", "relu", "leakyRelu"};
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_ACTIVATION_HPP_
