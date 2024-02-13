// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "network.cpp"

// namespace py = pybind11;

// PYBIND11_MODULE(neuralnetwork_module, m) {
//     py::class_<NeuralNetwork>(m, "NeuralNetwork")
//         .def(py::init<const std::vector<int>& )
// }


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(my_module, m) {
    m.doc() = "pybind11 example plugin";  // Optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}
