#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// Push constants for the collect (geometry) pass
struct UFScenePushConstants {
    glm::mat4 model;
    glm::mat4 viewProj;
    uint32_t  objectID;
    float     padding[3];  // align to 16 bytes
};

// Push constants for all fullscreen passes
struct UFFullscreenPushConstants {
    float    depthThreshold;
    float    colorThreshold;
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t targetLayer;    // Only used by resolve (ignored by others)
};

// Push constants for composite pass
struct UFCompositePushConstants {
    uint32_t numLayers;
    int32_t  displayLayer;   // -1 = composite all, 0-3 = single layer
};