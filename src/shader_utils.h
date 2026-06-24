#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace ShaderUtils {
    // Load a SPIR-V binary from file and create a VkShaderModule
    VkShaderModule createShaderModule(VkDevice device, const std::string& filepath);

    // Load raw bytes from file
    std::vector<char> readFile(const std::string& filepath);
}