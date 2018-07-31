/*
    Copyright IdeaLounge.io 2018
*/

#

namespace ncooper {
    namespace NN {
        template <class NeuronType>
        class Neuron {
            public:
                Neuron();
                ~Neuron();

                void forwardProp(const Vector<NeuronType>& inputVector);
                void activate();
            private:

                Vector<NeuronType>& weights; // arbitrary size
                int& bias;
        }
}
