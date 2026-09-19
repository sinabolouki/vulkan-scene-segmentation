#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Register Roots — Per-Root maxPosition via Direct Indexing
//
// For each pixel, walks the sorted fragment list and records the maximum
// list position at which each root appears. Uses rootMaxPos[root] with
// atomicMax — no fixed-size table, supports unlimited distinct roots.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX  0xFFFFFFFFu
#define MAX_FRAGS 32u

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    uint  unshadedColor;  // Packed RGBA8 (unlit base color)
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

layout(set = 0, binding = 4, std430) coherent buffer RootMaxPosBuffer {
    uint rootMaxPos[];
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

    uint idx = imageLoad(headPointerImage, coord).r;
    uint position = 0u;

    while (idx != NULL_IDX && idx < validCount && position < MAX_FRAGS) {
        uint root = parent[idx];

        if (root < validCount) {
            atomicMax(rootMaxPos[root], position);
        }

        position++;
        idx = nodes[idx].next;
    }
}