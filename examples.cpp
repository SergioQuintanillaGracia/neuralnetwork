#include <iostream>
#include <vector>
// TODO: Create network.h file and import that instead of network.cpp
#include "network.cpp"

void example1();
void example2();
void printResult(std::vector<double> vec);

int main() {
    example2();

    return 0;
}


void example1() {
    std::vector<int> input = {2, 4, 1};

    NeuralNetwork neuralNetwork(input, "./networks/examples/example1.weights", "./networks/examples/example1.bias");
    std::vector<double> inputLayer = {0.55, 0.9};

    std::vector<double> result = neuralNetwork.compute(inputLayer);
    printResult(result);

    GeneticNetworkTrainer trainer(&neuralNetwork, "./networks/examples/example1_mutations", 0.05, 0.05, 1);
    NeuralNetwork mutated1 = trainer.mutate(1, 1);

    result = mutated1.compute(inputLayer);
    printResult(result);

    NeuralNetwork mutated2 = trainer.mutate(1, 1);

    result = mutated2.compute(inputLayer);
    printResult(result);
}

void example2() {
    std::vector<int> input = {256, 128, 64, 1};

    NeuralNetwork neuralNetwork(input, "./networks/examples/example2.weights", "./networks/examples/example2.bias");
    GeneticNetworkTrainer trainer(&neuralNetwork, "./networks/examples/example1_mutations", 0.05, 0.05, 1);

    std::string obj1 = "Circle";
    std::string path1 = "./training/circles16x16/";
    std::string obj2 = "Square";
    std::string path2 = "./training/squares16x16/";
    trainer.trainBinary(obj1, path1, obj2, path2);
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}