#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.cpp"

void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> layers = {64, 32, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    std::string basePath = "./networks/squares_circles_8x8/64_32_1";
    std::string pointsPath = "1310_2000";
    NeuralNetwork* neuralNetwork = new NeuralNetwork(layers, basePath + "/progress/" + pointsPath + ".weights", basePath + "/progress/" + pointsPath + ".bias", false);
    
    GeneticNetworkTrainer trainer(neuralNetwork, basePath, 0, 0, 22);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles8x8/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares8x8/";
    trainer.trainBinary(obj1, path1, obj2, path2, 100000, 0.5, true, 1500);
    //std::cout << trainer.getFitness(neuralNetwork, path1, path2) << std::endl;
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}