#pragma once

#include <vector>
#include <string>

class Neuron {
public:
    double value;
    double bias;

    Neuron(double b = 0);

    void sigmoidActivation();
    void reluActivation();
};


class Layer {
private:
    std::vector<Neuron> neurons;
    std::vector<double> weights;

public:
    Layer* prev;
    Layer* next;
    int neuronAmount;

    Layer(int neuronCount, Layer* prevLayer = nullptr, Layer* nextLayer = nullptr);

    void loadWeights(const std::vector<double>& w);
    void loadBiases(const std::vector<double>& b);
    void setInitialValues(const std::vector<double>& initialValues);
    void computeValues();
    std::vector<double> getValues();
};


class NeuralNetwork {
private:
    std::vector<int> neuronsPerLayer;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    Layer* inputLayer;
    Layer* outputLayer;

    void initializeLayers();
    std::vector<std::vector<double>> getWeightsFromFile(const std::string& weightsPath);
    std::vector<std::vector<double>> getBiasesFromFile(const std::string& biasesPath);
    std::vector<std::vector<double>> generateRandomWeights(const std::vector<int>& neuronsPerLayer, double randomVariation);
    std::vector<std::vector<double>> generateRandomBiases(const std::vector<int>& neuronsPerLayer, double randomVariation);
    void assignWeights();
    void assignBiases();

public:
    NeuralNetwork(std::vector<int>& nPerLayer, const std::string& weightsPath, const std::string& biasesPath, bool randomize = false,
                  double randomWeightVariation = 0.3, double randomBiasVariation = 0.1);
    NeuralNetwork(std::vector<int>& nPerLayer, std::vector<std::vector<double>>&& weightsVec, std::vector<std::vector<double>>&& biasesVec);
    ~NeuralNetwork();

    void writeWeightsToFile(const std::string& weightsPath);
    void writeBiasesToFile(const std::string& biasesPath);
    std::vector<std::vector<double>>& getWeightsVector();
    std::vector<std::vector<double>>& getBiasesVector();
    std::vector<int> getNeuronsPerLayerVector();
    std::vector<double> compute(std::vector<double>& input);
    void printNeuronsData();
};


class GeneticNetworkTrainer {
private:
    bool baseNetworkIsOriginal;
    NeuralNetwork* baseNetwork;
    std::string trainPath;
    double weightMutationAmount;
    double biasMutationAmount;
    int mutationsPerGen;

    std::vector<std::vector<double>> mutateVector(std::vector<std::vector<double>>& vec, double mutationAmount, double rangeRandomness);
    NeuralNetwork* mutate(double rangeRandomness);

public:
    GeneticNetworkTrainer(NeuralNetwork* baseNet, const std::string& tPath, double wMutation, double bMutation, int mutations);

    double fitness(NeuralNetwork* network, const std::vector<std::string>& paths, int imageLimit);
    std::string getAccuracyString(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, int imageLimit = -1);
    std::vector<std::vector<std::vector<double>>> getFitnessData(NeuralNetwork* network, const std::vector<std::string>& paths, int imageLimit);
    void initializeCache(const std::vector<std::string>& paths);
    void train(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, double rangeRandomness, int currentGen,
                     bool writeNetworkData = false, bool multithread = true, bool enableOutput = false, int imageLimit = -1);
};