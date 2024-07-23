#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "network.h"
#include "os_tools.h"
#include "image_tools.h"

// Neuron class function definitions
Neuron::Neuron(double b) : bias(b) {};

void Neuron::sigmoidActivation() {
    // Apply the bias.
    value += bias;

    // Use the sigmoid function to set the new value of the neuron.
    value = 1 / (1 + std::exp(-value));
}

void Neuron::reluActivation() {
    // Apply the bias.
    value += bias;

    // Apply the ReLU activation function.
    value = std::max(0.0, value);
}

// Layer class function definitions
Layer::Layer(int neuronCount, Layer* prevLayer, Layer* nextLayer) : prev(prevLayer),
             next(nextLayer), neurons(neuronCount), neuronAmount(neuronCount) {}

void Layer::loadWeights(const std::vector<double>& w) {
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

void Layer::loadBiases(const std::vector<double>& b) {
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
        neurons[i].bias = b[i];
    }
}

void Layer::setInitialValues(const std::vector<double>& initialValues) {
    if (initialValues.size() != neuronAmount) {
        std::cerr << "Tried to initialize the values of a layer with a vector that is not the expected size "
                    << "(" << initialValues.size() << ", but should be " << neuronAmount << ". "
                    << "Values not initialized." << std::endl;
        return;
    }

    // Set initial values to the neurons of the layer.
    for (int i = 0; i < neuronAmount; i++) {
        neurons[i].value = initialValues[i];
    }
}

void Layer::computeValues() {
    // Computes the values of the neurons of this layer.
    if (prev == nullptr) {
        std::cerr << "Called computeValues from a layer with no previous layer. Values will not be computed." << std::endl;
        return;
    }

    // Reset current layer's neuron values.
    for (Neuron& n : neurons) {
        n.value = 0;
    }

    // Iterate through every neuron of the previous layer.
    for (size_t i = 0; i < prev->neuronAmount; i++) {
        double prevNeuronValue = prev->neurons[i].value;

        // Iterate through every weight and neuron of the current layer.
        for (size_t j = 0; j < neuronAmount; j++) {
            neurons[j].value += prevNeuronValue * weights[j + i * neuronAmount];
        }
    }

    for (Neuron& n : neurons) {
        if (next != nullptr) {
            n.reluActivation();
        } else {
            n.sigmoidActivation();
        }
    }
}

std::vector<double> Layer::getValues() {
    // Returns a vector containing the values of every neuron in this layer.
    std::vector<double> values;
    values.reserve(neurons.size());

    for (Neuron& n : neurons) {
        values.push_back(n.value);
    }

    return values;
}

std::vector<double> Layer::getValuesSoftmax() {
    // Returns the vector formed by applying softmax to the values of the neurons of the layer.
    // Softmax is calculated like this: exp(i) / (sum(exp(i)))
    double denominator = 0;

    std::vector<double> values;
    values.reserve(neurons.size());

    for (Neuron& n : neurons) {
        denominator += exp(n.value);
    }

    for (Neuron& n : neurons) {
        values.push_back(exp(n.value) / denominator);
    }

    return values;
}


// NeuralNetwork class function definitions
void NeuralNetwork::initializeLayers() {
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

std::vector<std::vector<double>> NeuralNetwork::getWeightsFromFile(const std::string& weightsPath) {
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

std::vector<std::vector<double>> NeuralNetwork::getBiasesFromFile(const std::string& biasesPath) {
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

std::vector<std::vector<double>> NeuralNetwork::generateRandomWeights(const std::vector<int>& neuronsPerLayer, double randomVariation) {
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

std::vector<std::vector<double>> NeuralNetwork::generateRandomBiases(const std::vector<int>& neuronsPerLayer, double randomVariation) {
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

void NeuralNetwork::assignWeights() {
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

void NeuralNetwork::assignBiases() {
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

NeuralNetwork::NeuralNetwork(std::vector<int>& nPerLayer, const std::string& weightsPath, const std::string& biasesPath, bool randomize,
                  double randomWeightVariation, double randomBiasVariation) : neuronsPerLayer(nPerLayer) {
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
        writeWeightsToFile(weightsPath);

        // Generate random biases.
        biases = generateRandomBiases(neuronsPerLayer, randomBiasVariation);
        writeBiasesToFile(biasesPath);
    } else {
        weights = getWeightsFromFile(weightsPath);
        biases = getBiasesFromFile(biasesPath);
    }

    assignWeights();
    assignBiases();
}

NeuralNetwork::NeuralNetwork(std::vector<int>& nPerLayer, std::vector<std::vector<double>>&& weightsVec, std::vector<std::vector<double>>&& biasesVec)
                : neuronsPerLayer(nPerLayer) {
    // This constructor will move weightsVec and biasesVec for optimization purposes.
    initializeLayers();

    weights = std::move(weightsVec);
    biases = std::move(biasesVec);

    assignWeights();
    assignBiases();
};

void NeuralNetwork::writeWeightsToFile(const std::string& weightsPath) {
    // Store the weights.
    std::ofstream fileW(weightsPath);

    if (!fileW.is_open()) {
        std::cerr << "Could not open weights file for writing: " << weightsPath << std::endl;
        return;
    }

    // Set a higher precision for storing the values.
    fileW << std::fixed << std::setprecision(20);

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

void NeuralNetwork::writeBiasesToFile(const std::string& biasesPath) {
    std::ofstream fileB(biasesPath);

    if (!fileB.is_open()) {
        std::cerr << "Could not open bias file for writing: " << biasesPath << std::endl;
        return;
    }

    // Set a higher precision for storing the values.
    fileB << std::fixed << std::setprecision(20);

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

NeuralNetwork::~NeuralNetwork() {
    // Delete every layer pointer.
    Layer* currLayer = inputLayer;
    
    while (currLayer != nullptr) {
        Layer* nextLayer = currLayer->next;
        delete currLayer;
        currLayer = nextLayer;
    }
}

std::vector<std::vector<double>>& NeuralNetwork::getWeightsVector() {
    return weights;
}

std::vector<std::vector<double>>& NeuralNetwork::getBiasesVector() {
    return biases;
}

std::vector<int> NeuralNetwork::getNeuronsPerLayerVector() {
    return neuronsPerLayer;
}

std::vector<double> NeuralNetwork::compute(std::vector<double>& input) {
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

void NeuralNetwork::printNeuronsData() {
    Layer* currLayer = inputLayer;

    while (currLayer != nullptr) {
        for (double d : currLayer->getValues()) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
        currLayer = currLayer->next;
    }
}


// GeneticNetworkTrainer class function definitions
std::vector<std::vector<double>> GeneticNetworkTrainer::mutateVector(std::vector<std::vector<double>>& vec, double mutationAmount, double rangeRandomness) {
    std::vector<std::vector<double>> mutatedVec;
    mutatedVec.reserve(vec.size());

    // Set up a time based seed and a random number generator.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Get a random number using normal distribution, that will be used to increase the mutation range a random amount.
    std::normal_distribution<double> randomRangeDistrib(0, rangeRandomness);
    double extraRandomness = randomRangeDistrib(generator);

    // Define the lower and upper bounds of the random number, taking into account extraRandomness.
    double lowerBound = -mutationAmount - extraRandomness;
    double upperBound = mutationAmount + extraRandomness;

    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);

    for (std::vector<double>& currVec : vec) {
        std::vector<double> currMut;
        currMut.reserve(currVec.size());

        for (double& d : currVec) {
            // Push back a randomly generated number with an uniform distribution.
            currMut.push_back(distribution(generator) + d);
        }

        mutatedVec.push_back(currMut);
    }

    return mutatedVec;
}

NeuralNetwork* GeneticNetworkTrainer::mutate(double rangeRandomness) {
    // Returns a mutated version of baseNetwork.

    // Get the mutated versions of the weights and biases in vector form.
    std::vector<std::vector<double>> mutatedWeights = mutateVector(baseNetwork->getWeightsVector(), weightMutationAmount, rangeRandomness);
    std::vector<std::vector<double>> mutatedBiases = mutateVector(baseNetwork->getBiasesVector(), biasMutationAmount, rangeRandomness);
    
    // Create and return the mutated NeuralNetwork.
    std::vector<int> layers = baseNetwork->getNeuronsPerLayerVector();
    NeuralNetwork* mutatedNetwork = new NeuralNetwork(layers, std::move(mutatedWeights), std::move(mutatedBiases));

    return mutatedNetwork;
}

GeneticNetworkTrainer::GeneticNetworkTrainer(NeuralNetwork* baseNet, const std::string& tPath, double wMutation, double bMutation, int mutations)
                        : baseNetwork(baseNet), baseNetworkIsOriginal(true), trainPath(tPath), weightMutationAmount(wMutation), biasMutationAmount(bMutation),
                        mutationsPerGen(mutations) {};

double GeneticNetworkTrainer::fitness(NeuralNetwork* network, const std::vector<std::string>& paths, int imageLimit) {
    std::vector<std::vector<std::vector<double>>> fitnessData = getFitnessData(network, paths, imageLimit);
    std::vector<int> correct;
    correct.resize(fitnessData.size());
    double points = 0;

    for (int i = 0; i < fitnessData.size(); i++) {
        // fitnessData[i] corresponds to the ith object's answer data.
        for (int j = 0; j < fitnessData[i].size(); j++) {
            // fitnessData[i][j] corresponds to a single ith object's image data, a vector with the value of each neuron after processing the image.
            // If fitnessData[i][j][i] is the greatest value of the vector, the network correctly classified the image.
            auto maxIt = std::max_element(fitnessData[i][j].begin(), fitnessData[i][j].end());

            if (std::distance(fitnessData[i][j].begin(), maxIt) == i) {
                correct[i]++;
                points++;
            }
        }
    }

    double max = correct[std::distance(correct.begin(), std::max_element(correct.begin(), correct.end()))];
    int min = correct[std::distance(correct.begin(), std::min_element(correct.begin(), correct.end()))];
    max = max != 0 ? max : 1;

    return points * (0.2 * (1 - (max - min) / max) + 0.8);
}

std::string GeneticNetworkTrainer::getAccuracyString(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, int imageLimit) {
    std::vector<std::vector<std::vector<double>>> fitnessData = getFitnessData(baseNetwork, paths, imageLimit);

    // Store the correct and total labeled images of each object i at its ith position.
    std::vector<double> correctCount(objNames.size(), 0);
    std::vector<int> totalCount(objNames.size(), 0);

    for (int i = 0; i < fitnessData.size(); i++) {
        totalCount[i] += fitnessData[i].size();
        // fitnessData[i] corresponds to the ith object's answer data.
        for (int j = 0; j < fitnessData[i].size(); j++) {
            // fitnessData[i][j] corresponds to a single ith object's image data, a vector with the value of each neuron after processing the image.
            // If fitnessData[i][j][i] is the greatest value of the vector, the network correctly classified the image.
            auto maxIt = std::max_element(fitnessData[i][j].begin(), fitnessData[i][j].end());

            if (std::distance(fitnessData[i][j].begin(), maxIt) == i) {
                correctCount[i]++;
            }
        }
    }

    std::string accuracyString = "";
    
    for (int i = 0; i < correctCount.size(); i++) {
        accuracyString.append(objNames[i] + ": " + std::to_string(correctCount[i] / totalCount[i] * 100) + "% | ");
    }

    accuracyString.append("General Accuracy: " + std::to_string(static_cast<double>(std::accumulate(correctCount.begin(), correctCount.end(), 0))
                          / std::accumulate(totalCount.begin(), totalCount.end(), 0) * 100) + "%");

    return accuracyString;
}
    
std::vector<std::vector<std::vector<double>>> GeneticNetworkTrainer::getFitnessData(NeuralNetwork* network, const std::vector<std::string>& paths, int imageLimit) {
    // Returns a vector with the following structure:
    // {{obj1img1result, obj2img2result, ...}, {obj2img1.result, obj2img2.result, ...}, ...},
    // where objnimgmresult are vectors with the output values of every neuron in the output layer.
    std::vector<std::vector<std::vector<double>>> sampleData;

    for (const std::string& path : paths) {
        std::vector<std::string> imagePaths = getFiles(path, true);

        // Calculate how many images the NeuralNetwork should be tested with.
        // The limit will be set to imageLimit if imageLimit is > 0. If it's greater or equal
        // than the number of images, or negative, every image will be checked.
        int limit = imageLimit > 0 ? (imageLimit > imagePaths.size() ? imagePaths.size() : imageLimit) : imagePaths.size();

        std::vector<std::vector<double>> data;
        data.reserve(limit);

        // If imageLimit is greater than 0, imageLimit images will be processed.
        for (int i = 0; i < limit; i++) {
            std::vector<double> input = extractBrightness(imagePaths[i], true);
            std::vector<double> result = network->compute(input);

            data.push_back(result);
        }
        sampleData.push_back(data);
    }

    return sampleData;
}

void GeneticNetworkTrainer::initializeCache(const std::vector<std::string>& paths) {
    // Initialize the cache of files and images before any multithreaded operations to prevent race conditions.
    for (const std::string& path : paths) {
        initializeFilesCache(path);
    }
    std::cout << "Files cache initialized\n";
    
    for (const std::string& path : paths) {
        initializeImageCache(getFiles(path, true));
    }
    std::cout << "Image cache initialized\n";
}
    
void GeneticNetworkTrainer::train(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, double rangeRandomness,
                                        int currentGen, bool writeNetworkData, bool multithread, bool enableOutput, int imageLimit) {
    // Trains a network to distinguish between 2 objects.
    // Images of the first object are analised from path1, and images of the second object from path2.

    std::vector<NeuralNetwork*> networkVector;
    std::vector<double> networkPoints;

    // Add the current base network to compare it to its mutations.
    networkVector.push_back(baseNetwork);
    
    // Fill the networkVector with mutationsPerGen - 1 mutations of the base network.
    for (int i = 1; i < mutationsPerGen; i++) {
        networkVector.push_back(mutate(rangeRandomness));
    }


    // Get the fitness of all NeuralNetworks and store it in points.
    if (multithread) {
        // Reserve the necessary space in the networkPoints vector.
        networkPoints.resize(networkVector.size());

        std::vector<std::thread> threads;
        std::mutex mutex;

        // Define the getFitnessThreaded lambda function that will be executed in each thread.
        auto getFitnessThreaded = [&](NeuralNetwork* nn, int index) {
            double points = fitness(nn, paths, imageLimit);
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
            networkPoints.push_back(fitness(nn, paths, imageLimit));
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
        if (!baseNetworkIsOriginal) {
            // If the base network is not the original, it must be deleted.
            delete baseNetwork;
        }
        baseNetwork = networkVector[maxPointsIndex];
        baseNetworkIsOriginal = false;
    }

    // Delete the networks that aren't the best performer:
    for (int i = 1; i < networkVector.size(); i++) {
        if (i != maxPointsIndex) {
            delete networkVector[i];
        }
    }

    if (writeNetworkData) {
        // Write the weights and biases of the best NeuralNetwork to disk.
        std::string path = trainPath;
        std::string weightsPath = path + "/" + "gen" + std::to_string(currentGen) + ".weights";
        std::string biasesPath = path + "/" + "gen" + std::to_string(currentGen) + ".bias";
        makeDir(path);

        baseNetwork->writeWeightsToFile(weightsPath);
        baseNetwork->writeBiasesToFile(biasesPath);
    }

    if (enableOutput) {
        std::cout << "Gen " << currentGen << " best points: " << maxPoints << '\n';
    }
}