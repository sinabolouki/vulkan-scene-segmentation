#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Init Parents Pass
//
// Walks the linked list for this pixel and sets parent[i] = i for each
// allocated fragment. This makes every fragment its own root (singleton tree).
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX  0xFFFFFFFF
#define MAX_FRAGS 32

// ─── Descriptors ─────────────────────────────────────────────────────────────

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  normalXY;    // Packed view-space normal (2x float16 via packHalf2x16)
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) buffer ParentBuffer {
    uint parent[];
};

// ─── Push Constants ──────────────────────────────────────────────────────────

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
};

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        parent[idx] = idx;  // each fragment is its own root
        idx = nodes[idx].next;
        count++;
    }
}