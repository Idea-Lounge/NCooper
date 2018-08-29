/*
    Copyright IdeaLounge.io 2018
 */

#ifndef NCOOPER_AI_NN_PREACTIVATION_LAYER_HPP_
#define NCOOPER_AI_NN_PREACTIVATION_LAYER_HPP_

#include <time.h>
#include <math.h>
#include <iostream>
#include <random>
#include <vector>

#include "ai/nn/layer/layer.hpp"
#include "math/linalg/matrix.hpp"
#include "math/linalg/vector.hpp"
#include "ai/nn/neuron.hpp"

namespace ncooper {
namespace ai {
namespace nn {
template <class DataType>
class PreActivationLayer : public Layer<DataType> {
 public:
    PreActivationLayer(int inputVectorSize, int outputVectorSize);
    ~PreActivationLayer();

    void randomizeWsAndBs();
    void randomizeWsAndBs(DataType start, DataType end);
    void loadWeights();

    void forwardProp(const ncooper::math::linalg::Vector<DataType>& inputVector);

    Neuron<DataType>& getNeuron(int index);
    const ncooper::math::linalg::Matrix<DataType>& getWeightsMatrix();
    const ncooper::math::linalg::Vector<DataType>& getBiasVector();

 protected:
    std::vector<Neuron<DataType> > neurons;
    ncooper::math::linalg::Matrix<DataType> weightsMatrix;
    ncooper::math::linalg::Vector<DataType> biasVector;
};
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

#endif  // NCOOPER_AI_NN_PREACTIVATION_LAYER_HPP_
