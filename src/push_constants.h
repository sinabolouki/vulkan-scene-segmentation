#pragma once

#include <glm/glm.hpp>

// Push constants sent to vertex + fragment shader
struct ScenePushConstants {
    glm::mat4 model;      // 64 bytes
    glm::mat4 viewProj;   // 64 bytes
    uint32_t objectID;    // 4 bytes
    float nearPlane;      // 4 bytes 
    float farPlane;       // 4 bytes 
    float padding;        // 4 bytes
};