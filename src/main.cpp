#include <iostream>
#include <vector>
#include "network.h"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {784, 128, 64, 3};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "../networks/0_1_2";
    std::string pointsPath = "gen100";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias", false);
    
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0, 0.1, 18);

    std::vector<std::string> objNames = {"0", "1", "2"};
    std::vector<std::string> paths = {"../training/mnist_png/training/0", "../training/mnist_png/training/1", "../training/mnist_png/training/2"};

    trainer.initializeCache(paths);
    // std::cout << trainer.getAccuracyString(objNames, paths) << std::endl;

    for (int i = 0; i < 100; i++) {
        trainer.train(objNames, paths, 0.15, i + 1, (i + 1) % 20 == 0, true, -1);
    }
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}