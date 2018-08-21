/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_LAYER_HPP_
#define NCOOPER_LAYER_HPP_

#include <iostream>
#include <random>
#include <time.h>
#include <math.h>

#include "math/linalg/matrix.hpp"
#include "math/linalg/vector.hpp"
#include "ai/nn/neuron.hpp"
// #include "ai/nn/activation.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class LayerType>
class Layer {
 public:
    Layer(int inputVectorSize, int numOfNeurons);
    ~Layer();

    void randomizeWsAndBs();
    void randomizeWsAndBs(LayerType start, LayerType end);
    void loadWeights();

    void forwardProp(const ncooper::math::linalg::Vector<LayerType>& inputVector);

    int getInputVectorSize();
    Neuron<LayerType>& getNeuron(int index);
    int getNumOfNeurons();
    const ncooper::math::linalg::Matrix<LayerType>& getWeightsMatrix();
    const ncooper::math::linalg::Vector<LayerType>& getBiasVector();
    const ncooper::math::linalg::Vector<LayerType>& getOutputVector();

 protected:
    std::vector<Neuron<LayerType> > neurons;
    ncooper::math::linalg::Matrix<LayerType> weightsMatrix;
    ncooper::math::linalg::Vector<LayerType> biasVector;
    ncooper::math::linalg::Vector<LayerType> outputVector;
    // Activation<LayerType> activation;
    int numOfNeurons;
    int inputVectorSize;
};
}
}
}

#endif  // NCOOPER_LAYER_HPP_
