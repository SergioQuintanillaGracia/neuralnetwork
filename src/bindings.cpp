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

void initializeCache(std::string path1, std::string path2) {
    if (trainer) {
        trainer->initializeCache(path1, path2);
    } else {
        std::cerr << "Could not initialize cache, as no GeneticNetworkTrainer has been initialized yet.\n";
    }
}

void initializeTrainer(std::string& trainPath, double wMutation, double bMutation, int mutations) {
    if (trainer) {
        delete trainer;
        trainer = nullptr;
    }

    trainer = new GeneticNetworkTrainer(neuralNetwork, trainPath, wMutation, bMutation, mutations);
    std::cout << "GeneticNetworkTrainer initialized\n";
}

void trainModel(std::string& obj1, std::string& path1, std::string& obj2, std::string& path2, double rangeRandomness,
                int fitnessFunctionID, int currentGen, bool writeNetworkData, bool multithread = true, int imageLimit = -1) {
    double (GeneticNetworkTrainer::*fitnessFunction)(NeuralNetwork*, const std::string&, const std::string&, int);

    if (trainer) {
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

        std::cout << "Before trainBinary\n";
        trainer->trainBinary(obj1, path1, obj2, path2, rangeRandomness, fitnessFunction, currentGen, writeNetworkData, multithread, imageLimit);
    } else {
        std::cerr << "No trainer has been initialized. Run initializeTrainer() before training the model.\n";
    }
}

PYBIND11_MODULE(bindings, m) {
    m.doc() = "NeuralNetwork Python bindings";

    m.def("loadModel", &loadModel, "Load a model with specified layers, weights, and biases");
    m.def("getModelAnswer", &getModelAnswer, "Get the model's answer for a given image path");
    m.def("initializeCache", &initializeCache, "Initialize image and files cache to avoid race conditions in multithreaded environments");
    m.def("initializeTrainer", &initializeTrainer, "Initialize the trainer for the model.");
    m.def("trainModel", &trainModel, "Train a model");
}