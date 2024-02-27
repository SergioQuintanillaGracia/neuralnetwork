#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_tools.h"
#include "network.h"

namespace py = pybind11;

NeuralNetwork* neuralNetwork = nullptr;
GeneticNetworkTrainer* trainer = nullptr;

void initializeModelFiles(std::vector<int> layers, std::string weightsPath, std::string biasesPath) {
    // Initializes a model's .weights and .bias files.
    // This function may be optimized in the future, adding the functionality to the NeuralNetwork class.
    neuralNetwork = new NeuralNetwork(layers, weightsPath, biasesPath, true);
    delete neuralNetwork;
    neuralNetwork = nullptr;
}

void loadModel(std::vector<int> layers, std::string weightsPath, std::string biasesPath) {
    if (neuralNetwork) {
        delete neuralNetwork;
        neuralNetwork = nullptr;
    }
    
    neuralNetwork = new NeuralNetwork(layers, weightsPath, biasesPath);
}

std::vector<double> getModelAnswer(std::string imagePath, bool useCache = false) {
    std::vector<double> input = extractBrightness(imagePath, useCache);
    return neuralNetwork->compute(input);
}

std::string getAccuracyString(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, int imageLimit = -1) {
    return trainer->getAccuracyString(objNames, paths, imageLimit);
}

void initializeCache(const std::vector<std::string>& paths) {
    if (trainer) {
        trainer->initializeCache(paths);
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

void trainModel(const std::vector<std::string>& objNames, const std::vector<std::string>& paths, double rangeRandomness,
                int currentGen, bool writeNetworkData, bool multithread = true, bool enableOutput = false,
                int imageLimit = -1) {

    if (trainer) {
        trainer->train(objNames, paths, rangeRandomness, currentGen, writeNetworkData, multithread, enableOutput, imageLimit);
    } else {
        std::cerr << "No trainer has been initialized. Run initializeTrainer() before training the model.\n";
    }
}

PYBIND11_MODULE(bindings, m) {
    m.doc() = "NeuralNetwork Python bindings";

    m.def("initializeModelFiles", &initializeModelFiles, "Initializes a model's .weights and .bias files");
    m.def("loadModel", &loadModel, "Load a model with specified layers, weights, and biases");
    m.def("getModelAnswer", &getModelAnswer, "Get the model's answer for a given image path");
    m.def("getAccuracyString", &getAccuracyString, "Get the accuracy string of the current GeneticNetworkTrainer base model.");
    m.def("initializeCache", &initializeCache, "Initialize image and files cache to avoid race conditions in multithreaded environments");
    m.def("initializeTrainer", &initializeTrainer, "Initialize the trainer for the model.");
    m.def("trainModel", &trainModel, "Train a model");
}