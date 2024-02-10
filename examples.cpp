#include <iostream>
#include <vector>
// TODO: Create network.h file and import that instead of network.cpp
#include "network.cpp"

void example1();
void example2();
void example3();
void printResult(std::vector<double> vec);

int main() {
    example3();

    return 0;
}


void example1() {
    Layer l1(2, nullptr);
    Layer l2(4, &l1);
    l1.next = &l2;
    Layer l3(1, &l2, nullptr);
    l2.next = &l3;

    NeuralNetwork neuralNetwork(&l1, &l3, "./weights/examples/example1.weights");

    std::vector<double> input;
    input.push_back(-0.55);
    input.push_back(0.9);

    std::vector<double> result = neuralNetwork.compute(input);
    printResult(result);
}

void example2() {
    Layer l1(2, nullptr);
    Layer l2(4, &l1);
    l1.next = &l2;
    Layer l3(1, &l2, nullptr);
    l2.next = &l3;

    NeuralNetwork neuralNetwork(&l1, &l3, "./weights/examples/example2.weights", true);

    std::vector<double> input;
    input.push_back(-0.55);
    input.push_back(0.9);

    std::vector<double> result = neuralNetwork.compute(input);
    printResult(result);
}

void example3() {
    Layer l1(64, nullptr);
    Layer l2(256, &l1);
    l1.next = &l2;
    Layer l3(512, &l2, nullptr);
    l2.next = &l3;
    Layer l4(1, &l3, nullptr);
    l3.next = &l4;

    NeuralNetwork neuralNetwork(&l1, &l4, "./weights/examples/example3.weights", true);

    std::vector<double> input = {
    -0.97, -0.27, 0.28, -0.76, -0.69, 0.68, 0.43, -0.95, 0.6, -0.8, 0.51, -0.05, 0.75, -0.81, -0.87, 0.99, 
    0.53, 0.84, 0.85, -0.02, -0.37, 0.67, 0.32, 0.44, 0.96, 0.55, -0.7, 0.35, 0.57, 0.58, -0.9, -0.33, 
    0.31, -0.99, -0.57, 0.31, 0.45, -0.56, -0.82, 0.12, 0.09, 0.13, -0.87, -0.35, -0.6, -0.95, 0.0, 0.16, 
    0.0, -0.92, 0.51, 0.29, -0.04, -0.59, -0.29, 0.42, 0.97, 0.37, -0.15, -0.4, -0.47, -0.41, 0.52, 0.58
};

    std::vector<double> result = neuralNetwork.compute(input);
    printResult(result);
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}