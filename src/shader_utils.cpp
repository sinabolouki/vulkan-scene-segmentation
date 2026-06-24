#include "shader_utils.h"
#include <fstream>
#include <iostream>

namespace ShaderUtils {

std::vector<char> readFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::string& filepath) {
    std::vector<char> code = readFile(filepath);

    if (code.empty()) {
        std::cerr << "Shader code is empty for file: " << filepath << std::endl;
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module from: " << filepath << std::endl;
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}

} // namespace ShaderUtils