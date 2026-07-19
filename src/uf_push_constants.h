#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// Push constants for the collect (geometry) pass
// Total: 64 + 64 + 64 + 4 + 4 + 4 + 4 = 204 bytes (padded to 208)
struct UFScenePushConstants {
    glm::mat4 model;       // 64 bytes
    glm::mat4 viewProj;    // 64 bytes
    glm::mat4 view;        // 64 bytes  ← NEW: needed for view-space normals
    uint32_t  objectID;    // 4 bytes
    float     nearPlane;   // 4 bytes
    float     farPlane;    // 4 bytes
    float     padding;     // 4 bytes (align to 16)
};
// static_assert(sizeof(UFScenePushConstants) == 208)

// Push constants for all fullscreen passes
struct UFFullscreenPushConstants {
    float    depthThreshold;
    float    colorThreshold;
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t targetLayer;    // Only used by resolve (ignored by others)
    float    nearPlane;      // ← NEW: needed for merge pass depth-to-viewspace
    float    farPlane;       // ← NEW: needed for merge pass depth-to-viewspace
    float    projX;          // ← NEW: projection[0][0] (focal length X)
    float    projY;          // ← NEW: projection[1][1] (focal length Y)
    float    pad0;           // align to multiple of 4
};

// Push constants for composite pass
struct UFCompositePushConstants {
    uint32_t numLayers;
    int32_t  displayLayer;   // -1 = composite all, 0-3 = single layer
};