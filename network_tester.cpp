#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.cpp"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {144, 100, 60, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "./networks/squares_circles_12x12/144_100_60_1";
    std::string pointsPath = "404";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias");
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0.02, 0.02, 18);

    bool validate = false;
    std::string obj1 = "Circle";
    std::string path1 = validate ? "./training/circles12x12_validation/" : "./training/circles12x12/";
    std::string obj2 = "Square";
    std::string path2 = validate ? "./training/squares12x12_validation/" : "./training/squares12x12/";

    std::cout << trainer.getAccuracyString(obj1, path1, obj2, path2) << std::endl;
}