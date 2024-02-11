#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.cpp"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> input = {64, 32, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    NeuralNetwork* neuralNetwork = new NeuralNetwork(input, "./networks/squares_circles_8x8/64_32_1/progress/fitnessBasic/gen8799.weights", "./networks/squares_circles_8x8/64_32_1/progress/fitnessBasic/gen8799.bias");
    GeneticNetworkTrainer trainer(neuralNetwork, "./networks/squares_circles_8x8/64_32_1", 0.05, 0.05, 18);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles8x8/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares8x8/";

    std::cout << trainer.getAccuracyString(obj1, path1, obj2, path2) << std::endl;
}