#pragma once

#include <glm/glm.hpp>

// Push constants sent to vertex + fragment shader
struct ScenePushConstants {
    glm::mat4 model;
    glm::mat4 viewProj;
    uint32_t objectID;    // For per-object identification
    float padding[3];     // Align to 16 bytes
};