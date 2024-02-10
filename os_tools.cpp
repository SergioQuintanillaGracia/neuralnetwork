#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void makeDir(const std::string& path) {
    // Create the directories.
    if (!fs::exists(path) && !fs::create_directories(path)) {
        std::cerr << "Could not create directory: " << path << std::endl;
    }
}

std::vector<std::string> getFiles(const std::string& path) {
    std::vector<std::string> paths;

    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string imgPath = entry.path().string();
            paths.push_back(imgPath);
        }
    }

    return paths;
}