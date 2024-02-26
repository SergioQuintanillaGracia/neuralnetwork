#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static std::unordered_map<std::string, std::vector<std::string>> filesCache;

void makeDir(const std::string& path);
std::vector<std::string> getFiles(const std::string& path, bool useCache = false);
void initializeFilesCache(const std::string& path);