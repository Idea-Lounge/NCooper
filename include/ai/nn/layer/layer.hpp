/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_AI_NN_LAYER_HPP_
#define NCOOPER_AI_NN_LAYER_HPP_

#include <time.h>
#include <math.h>

#include <iostream>
#include <random>
#include <vector>

#include "math/linalg/vector.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
class Layer {
 public:
    Layer(int inputVectorSize, int outputVectorSize);
    ~Layer();

    virtual void forwardProp(const ncooper::math::linalg::Vector<DataType>& inputVector);

    const int getInputVectorSize();
    const int getOutputVectorSize();
    const ncooper::math::linalg::Vector<DataType>& getOutputVector();

 protected:
    ncooper::math::linalg::Vector<DataType> outputVector;
    int outputVectorSize;
    int inputVectorSize;
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_AI_NN_PREACTIVATION_LAYER_HPP_
