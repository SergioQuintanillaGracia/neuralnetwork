cmake_minimum_required(VERSION 3.25.1)
project(NeuralNetwork)

# Add pybind 11 subdirectory
add_subdirectory(external/pybind11)

# Define the Python module
pybind11_add_module(network_binding, network_binding.cpp)
