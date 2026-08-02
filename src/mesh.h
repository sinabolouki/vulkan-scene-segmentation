#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributes(4);

        attributes[0].binding = 0;
        attributes[0].location = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);

        attributes[1].binding = 0;
        attributes[1].location = 1;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = offsetof(Vertex, normal);

        attributes[2].binding = 0;
        attributes[2].location = 2;
        attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[2].offset = offsetof(Vertex, color);

        attributes[3].binding = 0;
        attributes[3].location = 3;
        attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
        attributes[3].offset = offsetof(Vertex, uv);

        return attributes;
    }
};

// Material info extracted from MTL
struct MaterialInfo {
    std::string name;
    glm::vec3 diffuseColor = glm::vec3(1.0f);  // Kd
    std::string diffuseTexture;                  // map_Kd filename (empty = no texture)
    int32_t textureIndex = -1;                   // Index into texture array (-1 = none)
};

// Represents one object within the OBJ file
struct SceneObject {
    std::string name;
    uint32_t objectID;
    uint32_t indexOffset;
    uint32_t indexCount;
    int32_t  materialIndex = -1;    // Index into Scene::m_materials
    int32_t  textureIndex = -1;     // Resolved texture index for GPU
    glm::vec3 diffuseColor = glm::vec3(1.0f);
};