#pragma once

#include <string>
#include <vector>

bool LoadBinaryPngMask(const std::string& path,
                       int target_width,
                       int target_height,
                       std::vector<double>* mask,
                       std::string* error,
                       bool preserve_aspect = false,
                       int* source_width = nullptr,
                       int* source_height = nullptr);
