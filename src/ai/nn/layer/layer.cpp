/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/layer/layer.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class LayerType>
Layer<LayerType>::Layer(int numOfNeurons, int inputVectorSize) : numOfNeurons(numOfNeurons), inputVectorSize(inputVectorSize) {
    this->weightsMatrix = ncooper::math::linalg::Matrix<LayerType>(numOfNeurons, inputVectorSize);
    this->biasVector = ncooper::math::linalg::Vector<LayerType>(numOfNeurons, 0);
    this->outputVector = ncooper::math::linalg::Vector<LayerType>(numOfNeurons, 0);

    this->neurons.reserve(numOfNeurons);
    for (int i = 0; i < numOfNeurons; i++) {
        this->neurons.push_back(Neuron<LayerType>(this->weightsMatrix[i],
            this->biasVector[i],
            this->outputVector[i]));
    }
}

template <class LayerType>
Layer<LayerType>::~Layer() {

}

template <class LayerType>
void Layer<LayerType>::randomizeWsAndBs() {
    std::default_random_engine generator;
    generator.seed(time(NULL));

    std::uniform_real_distribution<LayerType> distribution(
        (LayerType) - 1 / sqrt(this->inputVectorSize),
        (LayerType) 1 / sqrt(this->inputVectorSize));

    for (int i = 0; i < this->weightsMatrix.getRows(); i++) {
        for (int j = 0; j < this->weightsMatrix.getCols(); j++) {
            this->weightsMatrix(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < this->biasVector.getSize(); i++) {
        this->biasVector[i] = distribution(generator);
    }
}

template <class LayerType>
void Layer<LayerType>::randomizeWsAndBs(LayerType start, LayerType end) {
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_real_distribution<LayerType> distribution(start, end);

    for (int i = 0; i < this->weightsMatrix.getRows(); i++) {
        for (int j = 0; j < this->weightsMatrix.getCols(); j++) {
            std::cout << distribution(generator) << std::endl;
            this->weightsMatrix(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < this->biasVector.getSize(); i++) {
        this->biasVector[i] = distribution(generator);
    }
}

template <>
void Layer<int>::randomizeWsAndBs(int start, int end) {
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_int_distribution<int> distribution(start, end);

    for (int i = 0; i < this->weightsMatrix.getRows(); i++) {
        for (int j = 0; j < this->weightsMatrix.getCols(); j++) {
            std::cout << distribution(generator) << std::endl;
            this->weightsMatrix(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < this->biasVector.getSize(); i++) {
        this->biasVector[i] = distribution(generator);
    }
}

template <class LayerType>
void Layer<LayerType>::forwardProp(const ncooper::math::linalg::Vector<LayerType> &inputVector) {
    ncooper::math::linalg::Vector<LayerType> preActivationVector = (this->weightsMatrix * inputVector) + this->biasVector;

    for (int i = 0; i < outputVector.getSize(); i++) {
        this->outputVector[i] = (1 / (1 + exp(-preActivationVector[i])));
    }
}

template <class LayerType>
const ncooper::math::linalg::Vector<LayerType>& Layer<LayerType>::getOutputVector() {
    return this->outputVector;
}

template <class LayerType>
Neuron<LayerType>& Layer<LayerType>::getNeuron(int index) {
    assert(index >= 0 && index < this->neurons.size());
    return this->neurons[index];
}

template <class LayerType>
int Layer<LayerType>::getNumOfNeurons() {
    return this->numOfNeurons;
}

template class Layer<int>;
template class Layer<float>;
}
}
}
