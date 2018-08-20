#include <iostream>
#include <math.h>
#include "ai/nn/layer/layer.hpp"
#include "math/linalg/vector.hpp"

int main() {
    try {
        ncooper::ai::nn::Layer<float> layer(3, 3);
        layer.randomizeWsAndBs();

        ncooper::math::linalg::Vector<float> inputVector({1, 3, 4});
        std::cout << "input: \n" << inputVector << std::endl;
        // layer.forwardProp(inputVector);
        std::cout << "lol " << std::endl;

        for (int i = 0; i < layer.getNumOfNeurons(); i++) {
            layer.getNeuron(i).forwardProp(inputVector);
            std::cout << layer.getNeuron(i).getOutput() << std::endl;
        }
        std::cout << "final output: " << layer.getOutputVector() << std::endl;

    } catch (const char* exception) {
        std::cout << exception << std::endl;
    }
}
