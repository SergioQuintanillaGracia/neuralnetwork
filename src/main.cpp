#include <iostream>
#include <vector>
#include "network.h"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {256, 96, 48, 2};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "../networks/circles_circumf_16x16/256_96_48_2";
    std::string pointsPath = "764";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias", false);
    
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0, 0.1, 18);

    std::vector<std::string> objNames = {"Circle", "Circumference"};
    std::vector<std::string> paths = {"../training/circles16x16_manual_aug/", "../training/empty_circles16x16_manual_aug/"};

    trainer.initializeCache(paths);
    std::cout << trainer.getAccuracyString(objNames, paths) << std::endl;

    // for (int i = 0; i < 500; i++) {
    //     trainer.train(objNames, paths, 0.15, i + 1, (i + 1) % 100 == 0, true, -1);
    // }
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}