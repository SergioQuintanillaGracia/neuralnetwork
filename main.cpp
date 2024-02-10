#include <iostream>
#include <vector>
// TODO: Create and import header files for image_tools and network.
#include "image_tools.cpp"
#include "network.cpp"

void example1();
void example2();
void example3();
void printResult(std::vector<double> vec);

int main() {
    // Create the layers of the neural network.
    Layer l1(256, nullptr);
    Layer l2(128, &l1);
    l1.next = &l2;
    Layer l3(64, &l2, nullptr);
    l2.next = &l3;
    Layer l4(1, &l3, nullptr);
    l3.next = &l4;

    NeuralNetwork neuralNetwork(&l1, &l4, "./networks/squares_circles_16x16/256_128_64_1/gen1/1/starting.weights",
        "./networks/squares_circles_16x16/256_128_64_1/gen1/1/starting.bias", true);

    std::vector<double> input = extractBrightness("./training/circles16x16/circles0003.png");
    std::vector<double> result = neuralNetwork.compute(input);
    printResult(result);

    return 0;
}

void printResult(std::vector<double> vec) {
    std::cout << "Result: ";
    for (double d : vec) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}