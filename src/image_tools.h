#pragma once

#include <string>
#include <unordered_map>
#include <vector>

std::vector<double> extractBrightness(const std::string& imagePath);
void initializeImageCache(const std::vector<std::string>& paths);