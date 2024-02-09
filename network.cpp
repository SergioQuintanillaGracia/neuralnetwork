#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

class Neuron {
    double value;

public:
    void setValue(double newValue) {
        value = newValue;
    }

    double getValue() {
        return value;
    }

    void normalizeValue(int enteringNeuronCount) {
        value /= enteringNeuronCount;
    }
};


class Layer {
    std::vector<Neuron> neurons;
    // Connection weights from the previous layer to this layer.
    // The first neurons.size() weights must be the connection weights between the first
    // neuron of the previous layer and each neuron of the current layer.
    std::vector<double> weights;

public:
    Layer* prev;
    Layer* next;

    Layer(int neuronCount, Layer* prevLayer = nullptr, Layer* nextLayer = nullptr) : prev(prevLayer), next(nextLayer) {
        neurons.reserve(neuronCount);

        // Fill the neurons vector.
        for (int i = 0; i < neuronCount; i++) {
            neurons.emplace_back();
        }
    }

    void loadWeights(std::vector<double>& w) {
        if (prev == nullptr) {
            std::cerr << "Tried to add weights to a layer with no previous layer. Weights will not be added." << std::endl;
            return;
        }

        int expectedSize = prev->neurons.size() * neurons.size();

        if (w.size() != expectedSize) {
            std::cerr << "Weights vector doesn't have the expected size. Weights not added. "
                      << "Expected: " << expectedSize << ", Size: " << w.size() << std::endl;
            return;
        }

        weights = w;
    }

    void setInitialValues(const std::vector<double>& initialValues) {
        if (initialValues.size() != neurons.size()) {
            std::cerr << "Tried to initialize the values of a layer with a vector that is not the expected size."
                      << "Values not initialized." << std::endl;
            return;
        }

        // Set initial values to the neurons of the layer.
        for (size_t i = 0; i < neurons.size(); i++) {
            neurons[i].setValue(initialValues[i]);
        }
    }

    void computeValues() {
        // Computes the values of the neurons of this layer.
        if (prev == nullptr) {
            std::cerr << "Called computeValues from a layer with no previous layer. Values will not be computed." << std::endl;
            return;
        }

        int prevNeuronAmount = prev->neurons.size();
        std::vector<Neuron>& prevNeurons = prev->neurons;

        // Iterate through every neuron of the previous layer.
        for (size_t i = 0; i < prevNeuronAmount; i++) {
            Neuron& prevNeuron = prevNeurons[i];

            // Iterate through every weight and neuron of the current layer.
            for (size_t j = 0; j < neurons.size(); j++) {
                double weight = weights[j + i * neurons.size()];
                neurons[j].setValue(neurons[j].getValue() + prevNeuron.getValue() * weight);
            }
        }

        // Normalize the value of each neuron of this layer.
        for (Neuron& n : neurons) {
            n.normalizeValue(prevNeuronAmount);
        }
    }

    std::vector<double> getValues() {
        // Returns a vector containing the values of every neuron in this layer.
        std::vector<double> values;

        for (Neuron& n : neurons) {
            values.push_back(n.getValue());
        }

        return values;
    }
};


class NeuralNetwork {
    Layer* inputLayer;
    Layer* outputLayer;
    std::vector<std::vector<double>> weights;

public:
    NeuralNetwork(Layer* inputL, Layer* outputL, const std::string& weightsPath) : inputLayer(inputL), outputLayer(outputL) {
        std::ifstream file(weightsPath);
        std::string line;
        std::vector<double> layerWeights;

        // Read the weightsPath file, extract the weights, and store them in the weights vector.
        if (file.is_open()) {
            while (getline(file, line)) {
                if (line == "#") {
                    // There is a change of layer.
                    weights.push_back(layerWeights);
                    layerWeights.clear();
                } else {
                    // Convert the line to a stream.
                    std::istringstream iss(line);
                    double weight;

                    // Read and store doubles from the stream into weight, and then into layerWeights.
                    while (iss >> weight) {
                        layerWeights.push_back(weight);
                    }
                }
            }

            // In case the file doesn't end with a #, push layerWeights to weights.
            if (!layerWeights.empty()) {
                weights.push_back(layerWeights);
            }

            file.close();

        } else {
            std::cerr << "Could not open weights file: " << weightsPath << std::endl;
        }

        // Iterate through every layer and assign them their weights.
        Layer* currLayer = inputLayer->next;

        for (std::vector<double>& w : weights) {
            if (currLayer == nullptr) {
                std::cerr << "The weights vector is too large for this neural network. Extra weights have been ignored. "
                          << "Weights vector size: " << weights.size() << std::endl;
                break;
            } else {
                currLayer->loadWeights(w);
                currLayer = currLayer->next;
            }
        }
    };

    std::vector<double> compute(std::vector<double> input) {
        // Load the input vector to the input layer.
        inputLayer->setInitialValues(input);

        // Compute the values through every layer until the outputLayer.
        Layer* currLayer = inputLayer->next;

        while (currLayer != nullptr) {
            currLayer->computeValues();
            currLayer = currLayer->next;
        }

        // Return the computed values.
        return outputLayer->getValues();
    }
};