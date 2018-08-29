
#include "ai/nn/network/network.hpp"

namespace ncooper {
namespace ai {
namespace nn {

template <class DataType>
Network<DataType>::Network(std::vector<int> networkArchitecture) :
    networkArchitecture(networkArchitecture) {
    std::cout << "initializing network" << std::endl;

    for (int i = 1; i < this->networkArchitecture.size(); i++) {
        this->hiddenLayers.push_back(new PreActivationLayer<DataType>(
                                         this->networkArchitecture[i - 1],
                                         this->networkArchitecture[i]));
        this->hiddenLayers.push_back(new ActivationLayer<DataType>(
                                         this->networkArchitecture[i],
                                         this->networkArchitecture[i]));
    }
}

template <class DataType>
Network<DataType>::~Network() {
    for (int i = 0; i < this->hiddenLayers.size(); i++) {
        delete this->hiddenLayers[i];
    }
}

template <class DataType>
void Network<DataType>::randomizeWsAndBs() {
    // loop assumes symmetric preactivation, activation pairing
    for (int i = 0; i < this->hiddenLayers.size(); i+=2) {
        dynamic_cast<PreActivationLayer<DataType>*>(this->hiddenLayers[i])->randomizeWsAndBs();
        std::cout << "Layer: " << i << std::endl;
        std::cout << "Weights Matrix: " << dynamic_cast<PreActivationLayer<DataType>*>(this->hiddenLayers[i])->getWeightsMatrix() << std::endl;
        std::cout << "Bias Vector: " << dynamic_cast<PreActivationLayer<DataType>*>(this->hiddenLayers[i])->getBiasVector() << std::endl;
    }
}

template <class DataType>
void Network<DataType>::forwardProp(const ncooper::math::linalg::Vector<DataType> &inputVector) {
    std::cout << "inputVectorSize: " << this->hiddenLayers[0]->getInputVectorSize() << std::endl;
    assert(this->hiddenLayers[0]->getInputVectorSize() == inputVector.getSize());
    ncooper::math::linalg::Vector<DataType> networkVector(inputVector);
    for (int i = 0; i < this->hiddenLayers.size(); i++) {
        std::cout << "Network Vector Layer " << i << ": " << networkVector << std::endl;
        this->hiddenLayers[i]->forwardProp(networkVector);
        networkVector = this->hiddenLayers[i]->getOutputVector();
    }
    this->outputVector = networkVector;
}

template <class DataType>
const std::vector<int>& Network<DataType>::getNetworkArchitecture() {
    return this->networkArchitecture;
}

template <class DataType>
Layer<DataType>& Network<DataType>::getLayer(int i) {
    return *this->hiddenLayers[i];
}

template <class DataType>
const ncooper::math::linalg::Vector<DataType>& Network<DataType>::getOutputVector() {
    return this->outputVector;
}

template class Network<int>;
template class Network<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
