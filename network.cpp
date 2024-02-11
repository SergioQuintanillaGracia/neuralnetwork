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

    void initializeLayers() {
        if (neuronsPerLayer.size() < 2) {
            std::cerr << "Cannot create a NeuralNetwork with less than 2 layers." << std::endl;
            return;
        }

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
    }

    std::vector<std::vector<double>> getWeightsFromFile(const std::string& weightsPath) {
        // Initialize the weights file for reading.
        std::ifstream fileW(weightsPath);
        std::string lineW;
        std::vector<std::vector<double>> weightsVec;
        std::vector<double> layerWeights;

        // Read the weightsPath file, extract the weights, and store them in the weightsVec vector.
        if (fileW.is_open()) {
            while (getline(fileW, lineW)) {
                if (lineW == "#") {
                    // There is a change of layer.
                    weightsVec.push_back(layerWeights);
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

            // In case the file doesn't end with a #, push layerWeights to weightsVec.
            if (!layerWeights.empty()) {
                weightsVec.push_back(layerWeights);
            }

            fileW.close();

        } else {
            std::cerr << "Could not open weights file: " << weightsPath << std::endl;
        }

        return weightsVec;
    }

    std::vector<std::vector<double>> getBiasesFromFile(const std::string& biasesPath) {
        // Initialize the biases file for reading.
        std::ifstream fileB(biasesPath);
        std::string lineB;
        std::vector<std::vector<double>> biasesVec;
        std::vector<double> layerBiases;

        // Read the biasesPath file, extract the biases, and store them in the biasesVec vector.
        if (fileB.is_open()) {
            while (getline(fileB, lineB)) {
                // Convert the line to a stream.
                std::istringstream iss(lineB);
                double bias;

                // Read and store doubles from the stream into bias, and then into layerBiases.
                while (iss >> bias) {
                    layerBiases.push_back(bias);
                }

                // Push the layerBiases vector to the biasesVec vector.
                biasesVec.push_back(layerBiases);
                layerBiases.clear();
            }

            fileB.close();

        } else {
            std::cerr << "Could not open biases file: " << biasesPath << std::endl;
        }

        return biasesVec;
    }

    std::vector<std::vector<double>> generateRandomWeights(const std::vector<int>& neuronsPerLayer, double randomVariation) {
        std::vector<std::vector<double>> weightsVec;

        // Initialize the random number generator, with a standard deviation of randomVariation.
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, randomVariation);

        // Iterate through each layer except for the last one.
        for (size_t i = 0; i < neuronsPerLayer.size() - 1; i++) {
            std::vector<double> layerWeights;
            int currentLayerSize = neuronsPerLayer[i];
            int nextLayerSize = neuronsPerLayer[i + 1];

            // Generate the weights for the connections from the current layer to the next one.
            for (size_t j = 0; j < currentLayerSize; j++) {
                for (size_t k = 0; k < nextLayerSize; k++) {
                    double randomWeight = std::max<double>(-1, std::min<double>(1, distribution(generator)));
                    layerWeights.push_back(randomWeight);
                }
            }

            weightsVec.push_back(layerWeights);
        }

        return weightsVec;
    }

    std::vector<std::vector<double>> generateRandomBiases(const std::vector<int>& neuronsPerLayer, double randomVariation) {
        std::vector<std::vector<double>> biasesVec;

        // Initialize the random number generator, with a standard deviation of randomVariation.
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, randomVariation);

        // Iterate through each layer except for the first one.
        for (size_t i = 1; i < neuronsPerLayer.size(); i++) {
            std::vector<double> layerBiases;
            int currentLayerSize = neuronsPerLayer[i];

            for (size_t j = 0; j < currentLayerSize; j++) {
                double randomBias = std::max<double>(-0.5, std::min<double>(0.5, distribution(generator)));
                layerBiases.push_back(randomBias);
            }

            biasesVec.push_back(layerBiases);
        }

        return biasesVec;
    }

    void assignWeights() {
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
    }

    void assignBiases() {
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
    }

public:
    NeuralNetwork(std::vector<int>& nPerLayer, const std::string& weightsPath, const std::string& biasesPath, bool randomize = false,
                  double randomWeightVariation = 0.3, double randomBiasVariation = 0.1) : neuronsPerLayer(nPerLayer) {
        initializeLayers();

        std::vector<std::vector<double>> weightsVec;
        std::vector<std::vector<double>> biasesVec;

        if (randomize) {
            // Store the amount of neurons of each layer in a vector.
            std::vector<int> neuronsPerLayer;

            Layer* currLayer = inputLayer;

            while (currLayer != nullptr) {
                neuronsPerLayer.push_back(currLayer->neuronAmount);
                currLayer = currLayer->next;
            }

            // Generate random weights.
            weights = generateRandomWeights(neuronsPerLayer, randomWeightVariation);

            // Generate random biases.
            biases = generateRandomBiases(neuronsPerLayer, randomBiasVariation);
        } else {
            weights = getWeightsFromFile(weightsPath);
            biases = getBiasesFromFile(biasesPath);
        }

        assignWeights();
        assignBiases();
    }

    NeuralNetwork(std::vector<int>& nPerLayer, std::vector<std::vector<double>>& weightsVec, std::vector<std::vector<double>>& biasesVec)
                  : neuronsPerLayer(nPerLayer) {
        initializeLayers();

        weights = weightsVec;
        biases = biasesVec;

        assignWeights();
        assignBiases();
    };

    void writeWeightsToFile(const std::string& weightsPath) {
        // Store the weights.
        std::ofstream fileW(weightsPath);

        if (!fileW.is_open()) {
            std::cerr << "Could not open weights file for writing: " << weightsPath << std::endl;
            return;
        }

        // Iterate through each subvector.
        for (size_t i = 0; i < weights.size(); i++) {
            int currentLayerSize = neuronsPerLayer[i];
            int nextLayerSize = neuronsPerLayer[i + 1];

            // Write the weights for the connections from the current layer to the next one.
            for (size_t j = 0; j < currentLayerSize; j++) {
                for (size_t k = 0; k < nextLayerSize; k++) {
                    double weight = weights[i][j * nextLayerSize + k];
                    fileW << weight << " ";
                }

                fileW << "\n";
            }

            fileW << "#\n";
        }

        fileW.close();
    }

    void writeBiasesToFile(const std::string& biasesPath) {
        std::ofstream fileB(biasesPath);

        if (!fileB.is_open()) {
            std::cerr << "Could not open bias file for writing: " << biasesPath << std::endl;
            return;
        }

        // Iterate through each layer except for the first one.
        for (size_t i = 1; i < neuronsPerLayer.size(); i++) {
            int currentLayerSize = neuronsPerLayer[i];

            for (size_t j = 0; j < currentLayerSize; j++) {
                double bias = biases[i - 1][j];
                fileB << bias << " ";
            }

            fileB << "\n";
        }

        fileB.close();
    }

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

    NeuralNetwork* mutate() {
        // Returns a mutated version of baseNetwork.

        // Get the mutated versions of the weights and biases in vector form.
        std::vector<std::vector<double>> mutatedWeights = mutateVector(baseNetwork->getWeightsVector(), weightMutationAmount);
        std::vector<std::vector<double>> mutatedBiases = mutateVector(baseNetwork->getBiasesVector(), biasMutationAmount);
        
        // Create and return the mutated NeuralNetwork.
        std::vector<int> layers = baseNetwork->getNeuronsPerLayerVector();
        NeuralNetwork* mutatedNetwork = new NeuralNetwork(layers, mutatedWeights, mutatedBiases);

        return mutatedNetwork;
    }

public:
    GeneticNetworkTrainer(NeuralNetwork* baseNet, const std::string& tPath, double wMutation, double bMutation, int mutations)
                          : baseNetwork(baseNet), trainPath(tPath), weightMutationAmount(wMutation), biasMutationAmount(bMutation),
                            mutationsPerGen(mutations) {};

    double fitnessBasic(NeuralNetwork* network, const std::string& path1, const std::string& path2) {
        // Returns the sum of the total images the NeuralNetwork got right.
        std::vector<std::vector<int>> fitnessData = getFitnessData(network, path1, path2);

        return fitnessData[0][1] + fitnessData[1][1];
    }

    double fitnessEqual(NeuralNetwork* network, const std::string& path1, const std::string& path2) {
        // Returns the sum of the total images the NeuralNetwork got right times the max of a number between
        // 0 and 1, and 0.5.
        // This number is greater when the number of obj1 images the NeuralNetwork got right is similar to
        // the number of obj2 images it got right.
        // Use this function if you want the NeuralNetwork to learn to recognize obj1 and obj2 with a similar
        // precission.
        std::vector<std::vector<int>> fitnessData = getFitnessData(network, path1, path2);

        double obj1Total = fitnessData[0][0];
        double obj1Right = fitnessData[0][1];
        double obj2Total = fitnessData[1][0];
        double obj2Right = fitnessData[1][1];

        double ratio = std::max(0.5, std::min(obj1Right / (std::max(1.0, obj2Right)), obj2Right / (std::max(1.0, obj1Right))));
        double points = ratio * (obj1Right + obj2Right);

        return points;
    }

    std::string getAccuracyString(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2) {
        std::vector<std::vector<int>> fitnessData = getFitnessData(baseNetwork, path1, path2);

        double obj1Total = fitnessData[0][0];
        double obj1Right = fitnessData[0][1];
        double obj2Total = fitnessData[1][0];
        double obj2Right = fitnessData[1][1];

        std::string accuracyString = obj1 + ": " + std::to_string(obj1Right / obj1Total * 100) + "% | " + obj2 + ": "
                                     + std::to_string(obj2Right / obj2Total * 100) + "% | General Accuracy: "
                                     + std::to_string((obj1Right + obj2Right) / (obj1Total + obj2Total) * 100) + "%";

        return accuracyString;
    }
    
    std::vector<std::vector<int>> getFitnessData(NeuralNetwork* network, const std::string& path1, const std::string& path2) {
        // Returns a vector with the following structure:
        // {{totalSamples1, rightSamples1}, {totalSamples2, rightSamples2}}
        std::vector<std::vector<int>> sampleData;

        // Test against object 1 images.
        std::vector<std::string> paths1 = getFiles(path1);

        int totalSamples1 = paths1.size();
        int rightSamples1 = 0;

        for (std::string imagePath : paths1) {
            std::vector<double> input = extractBrightness(imagePath);
            double result = network->compute(input)[0];

            if (result <= 0.5) {
                // The neural network correctly identified the object.
                rightSamples1++;
            }
        }

        std::vector<int> sampleData1 = {totalSamples1, rightSamples1};
        sampleData.push_back(sampleData1);

        // Test against object 2 images.
        std::vector<std::string> paths2 = getFiles(path2);

        int totalSamples2 = paths2.size();
        int rightSamples2 = 0;

        for (std::string imagePath : paths2) {
            std::vector<double> input = extractBrightness(imagePath);
            double result = network->compute(input)[0];

            if (result > 0.5) {
                // The neural network correctly identified the object.
                rightSamples2++;
            }
        }

        std::vector<int> sampleData2 = {totalSamples2, rightSamples2};
        sampleData.push_back(sampleData2);

        return sampleData;
    }
    
    void trainBinary(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2, int genLimit, bool multithread = false) {
        // Trains a network to distinguish between 2 objects.
        // Images of the first object are analised from path1, and images of the second object from path2.
        double prevMaxPoints = -1;

        for (int gen = 1; gen <= genLimit; gen++) {
            std::vector<NeuralNetwork*> networkVector;
            std::vector<double> networkPoints;

            // Add the current base network to compare it to its mutations.
            networkVector.push_back(baseNetwork);
            
            // Fill the networkVector with mutationsPerGen - 1 mutations of the base network.
            for (int i = 1; i < mutationsPerGen; i++) {
                networkVector.push_back(mutate());
            }


            // Get the fitness of all NeuralNetworks and store it in points.
            if (multithread) {
                // Reserve the necessary space in the networkPoints vector.
                networkPoints.resize(networkVector.size());

                std::vector<std::thread> threads;
                std::mutex mutex;

                // Define the getFitnessThreaded lambda function that will be executed in each thread.
                auto getFitnessThreaded = [&](NeuralNetwork* nn, int index) {
                    double points = fitnessBasic(nn, path1, path2);
                    std::lock_guard<std::mutex> lock(mutex);
                    networkPoints[index] = points;
                };

                // Run fitness calculations.
                for (int i = 0; i < networkVector.size(); i++) {
                    // Create a thread for each NeuralNetwork.
                    threads.emplace_back(getFitnessThreaded, networkVector[i], i);
                }

                // Wait for all threads to finish.
                for (auto& thread : threads) {
                    thread.join();
                }

            } else {
                for (NeuralNetwork* nn : networkVector) {
                    networkPoints.push_back(fitnessBasic(nn, path1, path2));
                }
            }

            // Find the best performing network, and set it as the base network.
            double maxPoints = -1;
            double maxPointsIndex = -1;

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

            // Write the weights and biases of the best NeuralNetwork to disk, only if it performed
            // better.
            if (maxPoints > prevMaxPoints) {
                std::string path = trainPath;
                std::string weightsPath = path + "/" + "gen" + std::to_string(gen) + ".weights";
                std::string biasesPath = path + "/" + "gen" + std::to_string(gen) + ".bias";
                makeDir(path);

                baseNetwork->writeWeightsToFile(weightsPath);
                baseNetwork->writeBiasesToFile(biasesPath);

                prevMaxPoints = maxPoints;
            }

            std::cout << "Gen " << gen << " best points: " << maxPoints << std::endl;
        }
    }
}; 