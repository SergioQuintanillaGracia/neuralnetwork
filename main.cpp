#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "network.cpp"

void example1();
void example2();
void example3();
void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    std::vector<int> input = {256, 128, 64, 1};

    // Create the NeuralNetwork and GeneticNetworkTrainer objects.
    NeuralNetwork* neuralNetwork = new NeuralNetwork(input, "./networks/squares_circles_16x16/256_128_64_1/progress/start.weights", "./networks/squares_circles_16x16/256_128_64_1/progress/start.bias");
    GeneticNetworkTrainer trainer(neuralNetwork, "./networks/squares_circles_16x16/256_128_64_1", 0.15, 0.15, 10);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles16x16/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares16x16/";
    trainer.trainBinary(obj1, path1, obj2, path2, 200, true);
    //std::cout << trainer.getFitness(neuralNetwork, path1, path2) << std::endl;
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}