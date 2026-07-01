#version 450

// ═══════════════════════════════════════════════════════════════════
// FIND GLOBAL MIN DEPTH PASS
//
// For each pixel, walks the sorted list past consumed fragments.
// The first unconsumed fragment's depth → atomicMin to state buffer.
// After this pass, state buffer holds the global minimum unconsumed depth.
// ═══════════════════════════════════════════════════════════════════

layout(location = 0) in vec2 fragUV;

layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 3, std430) buffer StateBuffer {
    uint globalMinUint;
};

layout(set = 0, binding = 5, r32f) uniform readonly image2D consumedDepthImage;

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
} pc;

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    uint headIndex = imageLoad(headPointerImage, pixel).r;
    if (headIndex == PPLL_NULL) return;

    float consumedDepth = imageLoad(consumedDepthImage, pixel).r;

    // Walk sorted list, find first unconsumed fragment
    uint idx = headIndex;
    int steps = 0;

    while (idx != PPLL_NULL && steps < MAX_WALK) {
        float d = nodes[idx].depth;

        if (d > consumedDepth) {
            // First unconsumed fragment → submit to global min
            atomicMin(globalMinUint, floatBitsToUint(d));
            return;
        }

        idx = nodes[idx].next;
        steps++;
    }
}