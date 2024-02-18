#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.h"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {256, 96, 48, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "../networks/circles_circumf_16x16/256_96_48_1_mixed";
    std::string pointsPath = "269_270";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias");
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0.02, 0.02, 18);

    bool validate = true;
    std::string obj1 = "Circle";
    std::string path1 = validate ? "../training/circles16x16_validation/" : "../training/circles16x16/";
    std::string obj2 = "Circumference";
    std::string path2 = validate ? "../training/empty_circles16x16_validation/" : "../training/empty_circles16x16/";

    std::cout << trainer.getAccuracyString(obj1, path1, obj2, path2) << std::endl;
}