/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_ACTIVATION_HPP_
#define NCOOPER_ACTIVATION_HPP_

#include <iostream>
#include <math>
#include <algorithm>

namespace ncooper {
namespace ai {
namespace nn {
class Activation {

public:
Activation();
Activation(std::string activationName);

private:
float computeActivation(float z);
float identity(float z);
float sigmoid(float z);
float tanh(float z);
float relu(float z);
float leakyRelu(float z);
std::string activationName;
const std::vector<std::string> activationsList = {"identity", "sigmoid", "tanh", "relu", "leakyRelu"};
};
}
}
}

#endif // NCOOPER_ACTIVATION_HPP_
