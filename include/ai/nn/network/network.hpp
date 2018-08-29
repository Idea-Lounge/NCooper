/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_NETWORK_HPP_
#define NCOOPER_NETWORK_HPP_

#include <time.h>
#include <math.h>

#include <iostream>
#include <random>
#include <vector>
#include <utility>

#include "math/linalg/matrix.hpp"
#include "ai/nn/layer/layer.hpp"
#include "ai/nn/layer/preActivationLayer.hpp"
#include "ai/nn/layer/activationLayer.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
class Network {
 public:
    Network(std::vector<int> networkArchitecture);
    ~Network();

    void randomizeWsAndBs();

    void forwardProp(const ncooper::math::linalg::Vector<DataType>& inputVector);

    const std::vector<int>& getNetworkArchitecture();
    Layer<DataType>& getLayer(int i);
    const ncooper::math::linalg::Vector<DataType>& getOutputVector();

 private:
    std::vector<int> networkArchitecture;
    std::vector<Layer<DataType>*> hiddenLayers;
    ncooper::math::linalg::Vector<DataType> outputVector;
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_NETWORK_HPP_
