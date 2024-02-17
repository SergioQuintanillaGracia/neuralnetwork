#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_tools.h"
#include "network.h"

namespace py = pybind11;

NeuralNetwork* neuralNetwork = nullptr;

int add(int i, int j) {
    return i + j;
}

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

PYBIND11_MODULE(bindings, m) {
    m.doc() = "NeuralNetwork Python bindings"; // optional module docstring

    m.def("loadModel", &loadModel, "Load a model with specified layers, weights, and biases");
    m.def("getModelAnswer", &getModelAnswer, "Get the model's answer for a given image path");
}