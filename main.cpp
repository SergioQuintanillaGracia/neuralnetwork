#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.cpp"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {256, 128, 64, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "./networks/squares_circles_16x16/256_128_64_1";
    std::string genPath = "0";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/gen" + genPath + ".weights", basePath + "/progress/gen" + genPath + ".bias", true);
    GeneticNetworkTrainer trainer(neuralNetwork, "./networks/squares_circles_8x8/64_32_1", 0.025, 0.025, 18);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles8x8/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares8x8/";
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