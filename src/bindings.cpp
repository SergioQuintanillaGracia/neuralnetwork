#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_tools.h"
#include "network.h"

namespace py = pybind11;

NeuralNetwork* neuralNetwork = nullptr;
GeneticNetworkTrainer* trainer = nullptr;

void loadModel(std::vector<int> layers, std::string weightsPath, std::string biasesPath) {
    if (neuralNetwork) {
        delete neuralNetwork;
        neuralNetwork = nullptr;
    }

    std::string basePath = "../networks/circles_circumf_16x16/256_96_48_1";
    std::string pointsPath = "2728.94_3000";
    neuralNetwork = new NeuralNetwork(layers, weightsPath, biasesPath);
}

std::vector<double> getModelAnswer(std::string imagePath, bool useCache = false) {
    std::vector<double> input = extractBrightness(imagePath, useCache);
    return neuralNetwork->compute(input);
}

void trainModel(std::string& trainPath, double wMutation, double bMutation, int mutations, std::string& obj1,
                std::string& path1, std::string& obj2, std::string& path2, int genLimit, double rangeRandomness,
                int fitnessFunctionID, bool multithread = true, int imageLimit = -1) {
    double (GeneticNetworkTrainer::*fitnessFunction)(NeuralNetwork*, const std::string&, const std::string&, int);

    if (trainer) {
        delete trainer;
        trainer = nullptr;
    }

    trainer = new GeneticNetworkTrainer(neuralNetwork, trainPath, wMutation, bMutation, mutations);

    switch (fitnessFunctionID) {
        case 1:
            fitnessFunction = &GeneticNetworkTrainer::fitnessBasic;
            break;
        case 2:
            fitnessFunction = &GeneticNetworkTrainer::fitnessEqual;
            break;
        case 3:
            fitnessFunction = &GeneticNetworkTrainer::fitnessPercentage;
            break;
        case 4:
            fitnessFunction = &GeneticNetworkTrainer::fitnessPercentageLinear;
            break;
        case 5:
            fitnessFunction = &GeneticNetworkTrainer::fitnessPercentageHybrid;
            break;
        default:
            std::cerr << "Fitness function ID does not exist. Defaulting to fitnessPercentageHybrid\n";
            fitnessFunction = &GeneticNetworkTrainer::fitnessPercentageHybrid;
            break;
    }

    trainer->trainBinary(obj1, path1, obj2, path2, genLimit, rangeRandomness, fitnessFunction, true);
}

int getCurrentGen() {
    if (trainer) {
        return trainer->currentGen;
    } else {
        return -1;
    }
}

PYBIND11_MODULE(bindings, m) {
    m.doc() = "NeuralNetwork Python bindings";

    m.def("loadModel", &loadModel, "Load a model with specified layers, weights, and biases");
    m.def("getModelAnswer", &getModelAnswer, "Get the model's answer for a given image path");
    m.def("trainModel", &trainModel, "Train a model");
    m.def("getCurrentGen", &getCurrentGen, "Get the current generation of the training of the model");
}