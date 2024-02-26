#include <iostream>
#include <vector>
#include "network.h"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {256, 96, 48, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "../networks/circles_circumf_16x16/256_96_48_1_mixed";
    std::string pointsPath = "test";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias", false);
    
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0, 0.1, 18);

    std::string obj1 = "Circle";
    std::string path1 = "../training/circles16x16_manual_aug/";
    std::string obj2 = "Circumference";
    std::string path2 = "../training/empty_circles16x16_manual_aug/";

    trainer.initializeCache(path1, path2);

    for (int i = 0; i < 500; i++) {
        trainer.train(obj1, path1, obj2, path2, 0.15, &GeneticNetworkTrainer::fitnessPercentageLinear, i + 1, (i + 1) % 100 == 0, true, -1);
    }
    //std::cout << trainer.getFitness(neuralNetwork, path1, path2) << std::endl;
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}