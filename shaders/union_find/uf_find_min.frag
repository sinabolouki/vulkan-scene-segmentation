
#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Find Global Minimum maxPosition
//
// Runs AFTER register_roots. Each pixel walks its fragment list, reads the
// (now-final) rootMaxPosition for each root, and atomicMin's it into a
// single global uint. The resolve pass then uses this value.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX 0xFFFFFFFFu
#define MAX_FRAGS 32u

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    float normalX;
    float normalY;
    float normalZ;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) readonly buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) readonly buffer ParentBuffer {
    uint parent[];
};

layout(set = 0, binding = 4, std430) readonly buffer RootMaxPosBuffer {
    uint rootMaxPos[];
};

layout(set = 0, binding = 5, std430) coherent buffer GlobalMinBuffer {
    uint globalMinMaxPos;
};

layout(push_constant) uniform PushConstants {
    float planeThreshold;
    float proximityScale;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
    float nearPlane;
    float farPlane;
    float projX;
    float projY;
    float colorThreshold;
};

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    uint validCount = min(counter, uint(rootMaxPos.length()));

    // Track which roots we've already submitted (avoid redundant atomics)
    // Simple approach: just submit the front fragment's root
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX || idx >= validCount) return;

    uint root = parent[idx];
    if (root < validCount) {
        uint maxPos = rootMaxPos[root];
        atomicMin(globalMinMaxPos, maxPos);
    }
}