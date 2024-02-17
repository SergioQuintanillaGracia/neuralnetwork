#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include "os_tools.h"

void makeDir(const std::string& path) {
    // Create the directories.
    if (!fs::exists(path) && !fs::create_directories(path)) {
        std::cerr << "Could not create directory: " << path << std::endl;
    }
}

std::vector<std::string> getFiles(const std::string& path, bool useCache) {

    if (useCache) {
        auto it = filesCache.find(path);

        if (it != filesCache.end()) {
            // The path has been found in the cache, return it.
            return it->second;
        }
    }

    std::vector<std::string> paths;

    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string imgPath = entry.path().string();
            paths.push_back(imgPath);
        }
    }

    if (useCache) {
        // Cache the results.
        filesCache[path] = paths;
    }

    return paths;
}

void initializeFilesCache(std::string& path) {
    getFiles(path, true);
}