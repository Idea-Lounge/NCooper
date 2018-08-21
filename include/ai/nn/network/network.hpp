/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_NETWORK_HPP_
#define NCOOPER_NETWORK_HPP_

#include <iostream>
#include <random>
#include <time.h>
#include <math.h>

#include "math/linalg/matrix.hpp"
#include "ai/nn/layer/layer.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class NetworkType>
class Network {
 public:
    Network(std::vector<std::pair<int, int>> networkArchitecture);
    ~Network();

    void randomizeWsAndBs();

    void forwardProp(const ncooper::math::linalg::Vector<NetworkType>& inputVector);

    const std::vector<std::pair<int, int>>& getNetworkArchitecture();
    Layer<NetworkType>& getLayer(int i);
    const ncooper::math::linalg::Vector<NetworkType>& getOutputVector();

 private:
    std::vector<std::pair<int, int>> networkArchitecture;
    std::vector<Layer<NetworkType>> hiddenLayers;
    ncooper::math::linalg::Vector<NetworkType> outputVector;
};
}
}
}

#endif  // NCOOPER_NETWORK_HPP_
