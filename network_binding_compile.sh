g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` network_binding.cpp -o network_binding.so -Iexternal/pybind11/include
