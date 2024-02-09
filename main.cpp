#include <iostream>
#include <vector>
// TODO: Create network.h file and import that instead of network.cpp
#include "network.cpp"

void example1();
void printResult(std::vector<double> vec);

int main() {
    example1();

    return 0;
}

void example1() {
    Layer l1(2, nullptr);
    Layer l2(4, &l1);
    l1.next = &l2;
    Layer l3(1, &l2, nullptr);
    l2.next = &l3;

    NeuralNetwork neuralNetwork(&l1, &l3, "./weights/examples/example1.weights");

    std::vector<double> exampleInput;
    exampleInput.push_back(-0.55);
    exampleInput.push_back(0.9);

    std::vector<double> result = neuralNetwork.compute(exampleInput);
    printResult(result);
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}