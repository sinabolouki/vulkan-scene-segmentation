#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// Push constants for the collect (geometry) pass
struct UFScenePushConstants {
    glm::mat4 model;       // 64 bytes
    glm::mat4 viewProj;    // 64 bytes
    glm::mat4 view;        // 64 bytes
    uint32_t  objectID;    // 4 bytes
    float     nearPlane;   // 4 bytes
    float     farPlane;    // 4 bytes
    float     padding;     // 4 bytes (align to 16)
};

// Push constants for all fullscreen passes
struct UFFullscreenPushConstants {
    float    planeThreshold;   // Tangent-plane residual tolerance (e.g. 1.5)
    float    proximityScale;   // Proximity fallback multiplier (e.g. 3.0)
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t targetLayer;
    float    nearPlane;
    float    farPlane;
    float    projX;
    float    projY;
    float    pad0;
};

// Push constants for composite pass
struct UFCompositePushConstants {
    uint32_t numLayers;
    int32_t  displayLayer;
};