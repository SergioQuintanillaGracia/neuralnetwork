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
    NeuralNetwork* baseNetwork;
    std::string trainPath;
    double weightMutationAmount;
    double biasMutationAmount;
    int mutationsPerGen;

    std::vector<std::vector<double>> mutateVector(std::vector<std::vector<double>>& vec, double mutationAmount, double rangeRandomness);
    NeuralNetwork* mutate(double rangeRandomness);

public:
    int currentGen = 0;

    GeneticNetworkTrainer(NeuralNetwork* baseNet, const std::string& tPath, double wMutation, double bMutation, int mutations);

    double fitnessBasic(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    double fitnessEqual(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    double fitnessPercentage(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    double fitnessPercentageLinear(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    // 10% of the points come from fitnessPercentage, and 90% from fitnessBasic.
    double fitnessPercentageHybrid(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    std::string getAccuracyString(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2, int imageLimit = -1);
    std::vector<std::vector<double>> getFitnessData(NeuralNetwork* network, const std::string& path1, const std::string& path2, int imageLimit);
    void trainBinary(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2, int genLimit, double rangeRandomness, double (GeneticNetworkTrainer::*fitnessFunction)(NeuralNetwork*, const std::string&, const std::string&, int), bool multithread = true, int imageLimit = -1);
};