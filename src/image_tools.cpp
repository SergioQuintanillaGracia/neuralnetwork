#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <unordered_map>
#include <vector>
#include "image_tools.h"
#include "libraries/stb_image.h"

std::unordered_map<std::string, std::vector<double>> imageCache;

std::vector<double> extractBrightness(const std::string& imagePath, bool useCache = true) {
    if (useCache) {
        // Check if the image has already been processed.
        auto it = imageCache.find(imagePath);
        if (it != imageCache.end()) {
            // The image is cached, return its brightness values.
            return it->second;
        }
    }

    // Returns a vector with the brightness values of each pixel of the image.
    // Load the image, width, height, and channels.
    int width, height, channels;
    // The 1 at the end converts the image to grayscale.
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 1);

    if (img == nullptr) {
        std::cerr << "Could not open or find the image: " << imagePath << std::endl;
        return {};
    }

    std::vector<double> brightnessValues;
    brightnessValues.reserve(width * height);

    // Store each pixel's brightness as a value between 0 and 1 into the brightnessValues vector.
    for (int i = 0; i < width * height; i++) {
        brightnessValues.push_back(img[i] / 255.0);
    }

    // Free the memory allocated by using stbi_load.
    stbi_image_free(img);

    if (useCache) {
        // Cache the results.
        imageCache[imagePath] = brightnessValues;
    }

    return brightnessValues;
}

void initializeImageCache(const std::vector<std::string>& paths) {
    // This function must be called for every path where training images are, BEFORE
    // any multithreaded operations.
    for (const std::string& path : paths) {
        extractBrightness(path, true);
    }
}