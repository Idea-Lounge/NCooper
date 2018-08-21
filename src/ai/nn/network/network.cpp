
#include "ai/nn/network/network.hpp"

namespace ncooper {
namespace ai {
namespace nn {

template <class NetworkType>
Network<NetworkType>::Network(std::vector<std::pair<int, int> > networkArchitecture) :
    networkArchitecture(networkArchitecture) {
    std::cout << "initializing network" << std::endl;
    for (int i = 0; i < networkArchitecture.size() - 1; i++) {
        std::cout << networkArchitecture[i].second << " : "
                  << networkArchitecture[i + 1].first
                  << std::endl;
        if (networkArchitecture[i].second != networkArchitecture[i + 1].first) {
            throw "Invalid network architecture!";
        }
    }

    std::cout << "checked network" << std::endl;
    for (int i = 0; i < this->networkArchitecture.size(); i++) {
        this->hiddenLayers.push_back(
            ncooper::ai::nn::Layer<NetworkType>(networkArchitecture[i].first,
                                                networkArchitecture[i].second)
            );
    }
}

template <class NetworkType>
Network<NetworkType>::~Network() {

}

template <class NetworkType>
void Network<NetworkType>::randomizeWsAndBs() {
    for (int i = 0; i < this->hiddenLayers.size(); i++) {
        this->hiddenLayers[i].randomizeWsAndBs();
        std::cout << "Layer: " << i << std::endl;
        std::cout << "Weights Matrix: " << this->hiddenLayers[i].getWeightsMatrix() << std::endl;
        std::cout << "Bias Vector: " << this->hiddenLayers[i].getBiasVector() << std::endl;
    }
}

template <class NetworkType>
void Network<NetworkType>::forwardProp(const ncooper::math::linalg::Vector<NetworkType> &inputVector) {
    std::cout << "inputVectorSize: " << this->hiddenLayers[0].getInputVectorSize() << std::endl;
    assert(this->hiddenLayers[0].getInputVectorSize() == inputVector.getSize());
    ncooper::math::linalg::Vector<NetworkType> networkVector(inputVector);
    for (int i = 0; i < this->hiddenLayers.size(); i++) {
        std::cout << "Network Vector Layer " << i << ": " << networkVector << std::endl;
        this->hiddenLayers[i].forwardProp(networkVector);
        networkVector = this->hiddenLayers[i].getOutputVector();
    }
    this->outputVector = networkVector;
}

template <class NetworkType>
const std::vector<std::pair<int, int> >& Network<NetworkType>::getNetworkArchitecture() {
    return this->networkArchitecture;
}

template <class NetworkType>
Layer<NetworkType>& Network<NetworkType>::getLayer(int i) {
    return this->hiddenLayers[i];
}

template <class NetworkType>
const ncooper::math::linalg::Vector<NetworkType>& Network<NetworkType>::getOutputVector() {
    return this->outputVector;
}

template class Network<int>;
template class Network<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
