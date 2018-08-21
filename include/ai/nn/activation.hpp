/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_ACTIVATION_HPP_
#define NCOOPER_ACTIVATION_HPP_

#include <iostream>
#include <math.h>
#include <algorithm>

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
}
}
}

#endif // NCOOPER_ACTIVATION_HPP_
