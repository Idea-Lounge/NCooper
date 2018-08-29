/*
    Copyright IdeaLounge.io 2018
 */

#include "ai/nn/layer/preActivationLayer.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
PreActivationLayer<DataType>::PreActivationLayer(int inputVectorSize, int outputVectorSize) :
    Layer<DataType>(inputVectorSize, outputVectorSize) {
    this->weightsMatrix = ncooper::math::linalg::Matrix<DataType>(outputVectorSize, inputVectorSize);
    this->biasVector = ncooper::math::linalg::Vector<DataType>(outputVectorSize);
    this->outputVector = ncooper::math::linalg::Vector<DataType>(outputVectorSize);

    this->neurons.reserve(outputVectorSize);
    for (int i = 0; i < outputVectorSize; i++) {
        this->neurons.push_back(Neuron<DataType>(this->weightsMatrix[i],
                                                 this->biasVector[i],
                                                 this->outputVector[i]));
    }
}

template <class DataType>
PreActivationLayer<DataType>::~PreActivationLayer() {

}

template <class DataType>
void PreActivationLayer<DataType>::randomizeWsAndBs() {
    std::default_random_engine generator;
    generator.seed(time(NULL));

    std::uniform_real_distribution<DataType> distribution(
        (DataType) - 1 / sqrt(this->inputVectorSize),
        (DataType) 1 / sqrt(this->inputVectorSize));

    for (int i = 0; i < this->weightsMatrix.getRows(); i++) {
        for (int j = 0; j < this->weightsMatrix.getCols(); j++) {
            this->weightsMatrix(i, j) = distribution(generator);
        }
    }
    for (int i = 0; i < this->biasVector.getSize(); i++) {
        this->biasVector[i] = distribution(generator);
    }
}

template <class DataType>
void PreActivationLayer<DataType>::randomizeWsAndBs(DataType start, DataType end) {
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_real_distribution<DataType> distribution(start, end);

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
void PreActivationLayer<int>::randomizeWsAndBs(int start, int end) {
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

template <class DataType>
void PreActivationLayer<DataType>::forwardProp(const ncooper::math::linalg::Vector<DataType> &inputVector) {
    assert(this->inputVectorSize == inputVector.getSize());
    this->outputVector = (this->weightsMatrix * inputVector) + this->biasVector;
}

template <class DataType>
const ncooper::math::linalg::Matrix<DataType>& PreActivationLayer<DataType>::getWeightsMatrix() {
    return this->weightsMatrix;
}

template <class DataType>
const ncooper::math::linalg::Vector<DataType>& PreActivationLayer<DataType>::getBiasVector() {
    return this->biasVector;
}

template <class DataType>
Neuron<DataType>& PreActivationLayer<DataType>::getNeuron(int index) {
    assert(index >= 0 && index < this->neurons.size());
    return this->neurons[index];
}

template class PreActivationLayer<int>;
template class PreActivationLayer<float>;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper
