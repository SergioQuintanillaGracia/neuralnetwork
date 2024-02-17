c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) bindings.cpp network.cpp os_tools.cpp image_tools.cpp -o bindings$(python3-config --extension-suffix)
