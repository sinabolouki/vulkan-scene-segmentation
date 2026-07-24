#version 450

// ═══════════════════════════════════════════════════════════════════
// SEED PASS
//
// For each pixel, check if first unconsumed fragment's depth is
// close to the global minimum. If yes → label this pixel as SEED.
// These seed pixels form the starting region for growing.
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

layout(set = 0, binding = 3, std430) readonly buffer StateBuffer {
    uint globalMinUint;
};

layout(set = 0, binding = 4, r32ui) uniform uimage2D labelImage;
layout(set = 0, binding = 5, r32f) uniform readonly image2D consumedDepthImage;

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float proximityScale;
} pc;

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    uint headIndex = imageLoad(headPointerImage, pixel).r;
    if (headIndex == PPLL_NULL) return;

    // No fragments found globally (shouldn't happen if we got here)
    if (globalMinUint == PPLL_NULL) return;

    float globalMinDepth = uintBitsToFloat(globalMinUint);
    float consumedDepth = imageLoad(consumedDepthImage, pixel).r;

    // Walk sorted list, find first unconsumed fragment
    uint idx = headIndex;
    int steps = 0;

    while (idx != PPLL_NULL && steps < MAX_WALK) {
        float d = nodes[idx].depth;

        if (d > consumedDepth) {
            // First unconsumed fragment found
            // Label if its depth is close to global minimum
            if (abs(d - globalMinDepth) <= pc.depthThreshold) {
                imageStore(labelImage, pixel, uvec4(1u, 0, 0, 0));
            }
            return;
        }

        idx = nodes[idx].next;
        steps++;
    }
}