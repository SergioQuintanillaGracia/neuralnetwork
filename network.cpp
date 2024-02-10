#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
// TODO: Create a heather file for os_tools.cpp and include that instead.
#include "os_tools.cpp"
// TODO: Create a heather file for image_tools.cpp and include that instead.
#include "image_tools.cpp"


class Neuron {
    double value;
    double bias;

public:
    Neuron(double b = 0) : bias(b) {};

    void setValue(double newValue) {
        value = newValue;
    }

    double getValue() {
        return value;
    }

    void setBias(double newBias) {
        bias = newBias;
    }

    double getBias() {
        return bias;
    }

    void sigmoidActivation() {
        // Apply the bias.
        value += bias;

        // Use the sigmoid function to set the new value of the neuron.
        value = 1 / (1 + std::exp(-value));
    }

    void reluActivation() {
        // Apply the bias.
        value += bias;

        // Apply the ReLU activation function.
        value = std::max(0.0, value);
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
    int neuronAmount;

    Layer(int neuronCount, Layer* prevLayer = nullptr, Layer* nextLayer = nullptr) : prev(prevLayer), next(nextLayer) {
        neurons.reserve(neuronCount);
        neuronAmount = neuronCount;

        // Fill the neurons vector.
        for (int i = 0; i < neuronCount; i++) {
            neurons.emplace_back();
        }
    }

    void loadWeights(std::vector<double>& w) {
        if (prev == nullptr) {
            std::cerr << "Tried to add weights to a layer with no previous layer. Weights can't and will not be added." << std::endl;
            return;
        }

        int expectedSize = prev->neuronAmount * neuronAmount;

        if (w.size() != expectedSize) {
            std::cerr << "Weights vector doesn't have the expected size. Weights not added. "
                      << "Expected: " << expectedSize << ", Size: " << w.size() << std::endl;
            return;
        }

        weights = w;
    }

    void loadBiases(const std::vector<double>& b) {
        if (prev == nullptr) {
            std::cerr << "Tried to add biases to a layer with no previous layer. The input layer cannot have biases." << std::endl;
            return;
        }

        if (b.size() != neuronAmount) {
            std::cerr << "Biases vector doesn't have the expected size. Biases not added. "
                      << "Expected: " << neuronAmount << ", Size: " << b.size() << std::endl;
            return;
        }

        // Set the bias value to each neuron of the layer.
        for (size_t i = 0; i < neurons.size(); i++) {
            neurons[i].setBias(b[i]);
        }
    }

    void setInitialValues(const std::vector<double>& initialValues) {
        if (initialValues.size() != neuronAmount) {
            std::cerr << "Tried to initialize the values of a layer with a vector that is not the expected size. "
                      << "Values not initialized." << std::endl;
            return;
        }

        // Set initial values to the neurons of the layer.
        for (size_t i = 0; i < neuronAmount; i++) {
            neurons[i].setValue(initialValues[i]);
        }
    }

    void computeValues() {
        // Computes the values of the neurons of this layer.
        if (prev == nullptr) {
            std::cerr << "Called computeValues from a layer with no previous layer. Values will not be computed." << std::endl;
            return;
        }

        // Set the values of the neurons of this layer to 0.
        for (Neuron& n : neurons) {
            n.setValue(0);
        }

        int prevNeuronAmount = prev->neuronAmount;
        std::vector<Neuron>& prevNeurons = prev->neurons;

        // Iterate through every neuron of the previous layer.
        for (size_t i = 0; i < prevNeuronAmount; i++) {
            Neuron& prevNeuron = prevNeurons[i];

            // Iterate through every weight and neuron of the current layer.
            for (size_t j = 0; j < neuronAmount; j++) {
                double weight = weights[j + i * neuronAmount];
                neurons[j].setValue(neurons[j].getValue() + prevNeuron.getValue() * weight);
            }
        }

        if (prev != nullptr && next != nullptr) {
            // Apply the ReLU activation function if it is a hidden layer.
            for (Neuron& n : neurons) {
                n.reluActivation();
            }
        } else if (next == nullptr) {
            // This is the output layer. Apply the sigmoid activation function.
            for (Neuron& n : neurons) {
                n.sigmoidActivation();
            }
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
    std::vector<int> neuronsPerLayer;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    Layer* inputLayer;
    Layer* outputLayer;

    void generateRandomWeights(const std::string& path, const std::vector<int> neuronsPerLayer, double randomVariation) {
        std::ofstream file(path);

        if (!file.is_open()) {
            std::cerr << "Could not open weights file for writing: " << path << std::endl;
            return;
        }

        // Initialize the random number generator, with a standard deviation of randomVariation.
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, randomVariation);

        // Iterate through each layer except for the last one.
        for (size_t i = 0; i < neuronsPerLayer.size() - 1; i++) {
            int currentLayerSize = neuronsPerLayer[i];
            int nextLayerSize = neuronsPerLayer[i + 1];

            // Generate the weights for the connections from the current layer to the next one.
            for (size_t j = 0; j < currentLayerSize; j++) {
                for (size_t k = 0; k < nextLayerSize; k++) {
                    double randomWeight = std::max<double>(-1, std::min<double>(1, distribution(generator)));
                    file << randomWeight << " ";
                }

                file << "\n";
            }

            file << "#\n";
        }

        file.close();
    }

    void generateRandomBiases(const std::string& path, const std::vector<int> neuronsPerLayer, double randomVariation) {
        std::ofstream file(path);

        if (!file.is_open()) {
            std::cerr << "Could not open bias file for writing: " << path << std::endl;
            return;
        }

        // Initialize the random number generator, with a standard deviation of randomVariation.
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, randomVariation);

        // Iterate through each layer except for the first one.
        for (size_t i = 1; i < neuronsPerLayer.size(); i++) {
            int currentLayerSize = neuronsPerLayer[i];

            for (size_t j = 0; j < currentLayerSize; j++) {
                double randomBias = std::max<double>(-0.5, std::min<double>(0.5, distribution(generator)));
                file << randomBias << " ";
            }

            file << "\n";
        }

        file.close();
    }

public:
    NeuralNetwork(std::vector<int>& nPerLayer, const std::string& weightsPath, const std::string& biasesPath, bool randomize = false,
                  double randomWeightVariation = 0.3, double randomBiasVariation = 0.1) {
        if (nPerLayer.size() < 2) {
            std::cerr << "Cannot create a NeuralNetwork with less than 2 layers." << std::endl;
            return;
        }

        neuronsPerLayer = nPerLayer;

        // Create the neural network layers.
        inputLayer = new Layer(neuronsPerLayer[0], nullptr, nullptr);
        Layer* prevLayer = inputLayer;
        Layer* currLayer;

        for (int i = 1; i < neuronsPerLayer.size(); i++) {
            currLayer = new Layer(neuronsPerLayer[i], prevLayer, nullptr);
            prevLayer->next = currLayer;
            prevLayer = currLayer;
        }

        outputLayer = currLayer;
        
        if (randomize) {
            // Store the amount of neurons of each layer in a vector.
            std::vector<int> neuronsPerLayer;

            Layer* currLayer = inputLayer;

            while (currLayer != nullptr) {
                neuronsPerLayer.push_back(currLayer->neuronAmount);
                currLayer = currLayer->next;
            }

            // Generate the random weights file.
            generateRandomWeights(weightsPath, neuronsPerLayer, randomWeightVariation);

            // Generate the random biases file.
            generateRandomBiases(biasesPath, neuronsPerLayer, randomBiasVariation);
        }
        
        // Initialize the weights file for reading.
        std::ifstream fileW(weightsPath);
        std::string lineW;
        std::vector<double> layerWeights;

        // Read the weightsPath file, extract the weights, and store them in the weights vector.
        if (fileW.is_open()) {
            while (getline(fileW, lineW)) {
                if (lineW == "#") {
                    // There is a change of layer.
                    weights.push_back(layerWeights);
                    layerWeights.clear();
                } else {
                    // Convert the line to a stream.
                    std::istringstream iss(lineW);
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

            fileW.close();

        } else {
            std::cerr << "Could not open weights file: " << weightsPath << std::endl;
        }

        // Iterate through every layer and assign them their weights.
        Layer* currLayerW = inputLayer->next;

        for (std::vector<double>& w : weights) {
            if (currLayerW == nullptr) {
                std::cerr << "The weights vector is too large for this neural network. Extra weights have been ignored. "
                          << "Weights vector size: " << weights.size() << std::endl;
                break;
            } else {
                currLayerW->loadWeights(w);
                currLayerW = currLayerW->next;
            }
        }

        // Initialize the biases file for reading.
        std::ifstream fileB(biasesPath);
        std::string lineB;
        std::vector<double> layerBiases;

        // Read the biasesPath file, extract the biases, and store them in the biases vector.
        if (fileB.is_open()) {
            while (getline(fileB, lineB)) {
                // Convert the line to a stream.
                std::istringstream iss(lineB);
                double bias;

                // Read and store doubles from the stream into weight, and then into layerBiases.
                while (iss >> bias) {
                    layerBiases.push_back(bias);
                }

                // Push the layerBiases vector to the biases vector.
                biases.push_back(layerBiases);
                layerBiases.clear();
            }

            fileB.close();

        } else {
            std::cerr << "Could not open biases file: " << biasesPath << std::endl;
        }

        // Iterate through every layer and assign them their biases.
        Layer* currLayerB = inputLayer->next;

        for (std::vector<double>& b : biases) {
            if (currLayerB == nullptr) {
                std::cerr << "The biases vector is too large for this neural network. Extra biases have been ignored. "
                          << "Biases vector size: " << biases.size() << std::endl;
                break;
            } else {
                currLayerB->loadBiases(b);
                currLayerB = currLayerB->next;
            }
        }
    };

    ~NeuralNetwork() {
        // Delete every layer pointer.
        Layer* currLayer = inputLayer;
        
        while (currLayer != nullptr) {
            Layer* nextLayer = currLayer->next;
            delete currLayer;
            currLayer = nextLayer;
        }
    }

    std::vector<std::vector<double>>& getWeightsVector() {
        return weights;
    }

    std::vector<std::vector<double>>& getBiasesVector() {
        return biases;
    }

    std::vector<int> getNeuronsPerLayerVector() {
        return neuronsPerLayer;
    }

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


class GeneticNetworkTrainer {
    NeuralNetwork* baseNetwork;
    std::string trainPath;
    double weightMutationAmount;
    double biasMutationAmount;
    int mutationsPerGen;

    std::vector<std::vector<double>> mutateVector(std::vector<std::vector<double>>& vec, double mutationAmount) {
        std::vector<std::vector<double>> mutatedVec;

        // Set up a time based seed and a random number generator.
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);

        for (std::vector<double>& currVec: vec) {
            std::vector<double> currMut;

            for (double& d : currVec) {

                double lowerBound = d - mutationAmount;
                double upperBound = d + mutationAmount;

                std::uniform_real_distribution<double> distribution(lowerBound, upperBound);

                // Push back a randomly generated number, centered at the original vector double.
                currMut.push_back(distribution(generator));
            }

            mutatedVec.push_back(currMut);
        }

        return mutatedVec;
    }

    NeuralNetwork* mutate(int generation, int networkNumber) {
        // Returns a mutated version of the given network.
        // base: Base network. The network will be copied, and the copy mutated.
        // generation: Int representing the current generation.
        // networkNumber: Int used to name the weights and biases files. Goes from 1 to mutationsPerGen.
        std::string path = trainPath + "/gen" + std::to_string(generation);
        std::string weightsPath = path + "/" + std::to_string(networkNumber) + ".weights";
        std::string biasesPath = path + "/" + std::to_string(networkNumber) + ".bias";
        makeDir(path);

        // Get the mutated versions of the weights and biases in vector form.
        std::vector<std::vector<double>> mutatedWeights = mutateVector(baseNetwork->getWeightsVector(), weightMutationAmount);
        std::vector<std::vector<double>> mutatedBiases = mutateVector(baseNetwork->getBiasesVector(), biasMutationAmount);

        // Store the vectors in files with the correct formatting.
        // Store the weights.
        std::ofstream fileW(weightsPath);

        if (!fileW.is_open()) {
            std::cerr << "Could not open weights file for mutation writing: " << weightsPath << std::endl;
        }

        // Iterate through each subvector.
        std::vector<int> baseNeuronsPerLayer = baseNetwork->getNeuronsPerLayerVector();

        for (size_t i = 0; i < mutatedWeights.size(); i++) {
            int currentLayerSize = baseNeuronsPerLayer[i];
            int nextLayerSize = baseNeuronsPerLayer[i + 1];

            // Write the weights for the connections from the current layer to the next one.
            for (size_t j = 0; j < currentLayerSize; j++) {
                for (size_t k = 0; k < nextLayerSize; k++) {
                    double weight = mutatedWeights[i][j * nextLayerSize + k];
                    fileW << weight << " ";
                }

                fileW << "\n";
            }

            fileW << "#\n";
        }

        fileW.close();

        // Store the biases.
        std::ofstream fileB(biasesPath);

        if (!fileB.is_open()) {
            std::cerr << "Could not open bias file for mutation writing: " << biasesPath << std::endl;
        }

        // Iterate through each layer except for the first one.
        for (size_t i = 1; i < baseNeuronsPerLayer.size(); i++) {
            int currentLayerSize = baseNeuronsPerLayer[i];

            for (size_t j = 0; j < currentLayerSize; j++) {
                double bias = mutatedBiases[i - 1][j];
                fileB << bias << " ";
            }

            fileB << "\n";
        }

        fileB.close();

        // Create and return the mutated NeuralNetwork.
        std::vector<int> layers = {baseNeuronsPerLayer};
        NeuralNetwork* mutatedNetwork = new NeuralNetwork(layers, weightsPath, biasesPath);

        return mutatedNetwork;
    }

public:
    GeneticNetworkTrainer(NeuralNetwork* baseNet, const std::string& tPath, double wMutation, double bMutation, int mutations)
                          : baseNetwork(baseNet), trainPath(tPath), weightMutationAmount(wMutation), biasMutationAmount(bMutation),
                            mutationsPerGen(mutations) {};
    
    double getFitness(NeuralNetwork* network, const std::string& path1, const std::string& path2) {
        double points = 0;

        // Test against object 1 images.
        for (std::string imagePath : getFiles(path1)) {
            std::vector<double> input = extractBrightness(imagePath);
            double result = network->compute(input)[0];

            if (result <= 0.5) {
                // The neural network correctly identified the object.
                points++;
            }
        }

        // Test against object 2 images.
        for (std::string imagePath : getFiles(path2)) {
            std::vector<double> input = extractBrightness(imagePath);
            double result = network->compute(input)[0];

            if (result > 0.5) {
                // The neural network correctly identified the object.
                points++;
            }
        }

        return points;
    }
    
    void trainBinary(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2, int genLimit, bool multithread = false) {
        // Trains a network to distinguish between 2 objects.
        // Images of the first object are analised from path1, and images of the second object from path2.
        for (int gen = 1; gen <= genLimit; gen++) {
            std::vector<NeuralNetwork*> networkVector;
            std::vector<int> networkPoints;
            std::vector<std::thread> threads;
            std::mutex mutex;
            // Add the current base network to compare it to its mutations.
            networkVector.push_back(baseNetwork);
            
            // Fill the networkVector with mutationsPerGen - 1 mutations of the base network.
            for (int i = 1; i < mutationsPerGen; i++) {
                networkVector.push_back(mutate(gen, i + 1));
            }

            // Get the fitness of all NeuralNetworks and store it in points.
            // TODO: Multithread this process when multithread is set to true. The threads used will be given by the variable mutationsPerGen.
            for (NeuralNetwork* nn : networkVector) {
                networkPoints.push_back(getFitness(nn, path1, path2));
            }

            // Find the best performing network, and set it as the base network.
            int maxPoints = -1;
            int maxPointsIndex = -1;

            for (int i = 0; i < networkPoints.size(); i++) {
                if (networkPoints[i] >= maxPoints) {
                    maxPoints = networkPoints[i];
                    maxPointsIndex = i;
                }
            }

            // If the best network is not the base network (with index 0), update baseNetwork.
            if (maxPointsIndex != 0) {
                baseNetwork = networkVector[maxPointsIndex];
            }

            // Delete the networks that aren't the best performer:
            for (int i = 0; i < networkVector.size(); i++) {
                if (i != maxPointsIndex) {
                    delete networkVector[i];
                }
            }

            std::cout << "Gen " << gen << " best performer: " << maxPointsIndex + 1 << ", Points: " << maxPoints << std::endl;
        }
    }
}; 