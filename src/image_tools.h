#pragma once

#include <string>
#include <unordered_map>
#include <vector>

std::vector<double> extractBrightness(const std::string& imagePath, bool useCache);
void initializeImageCache(const std::vector<std::string>& paths);