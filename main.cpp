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
    std::string pointsPath = "424";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias", false);
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0.02, 0.01, 18);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles12x12/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares12x12/";
    trainer.trainBinary(obj1, path1, obj2, path2, 10000, true);
    //std::cout << trainer.getFitness(neuralNetwork, path1, path2) << std::endl;
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}