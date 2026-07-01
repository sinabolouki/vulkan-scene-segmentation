#version 450

// ═══════════════════════════════════════════════════════════════════
// RESOLVE PASS (per layer)
//
// For each pixel:
//   If labeled → output first unconsumed fragment's color
//                AND update consumed depth to that fragment's depth
//   If not labeled → output transparent (alpha = 0)
// ═══════════════════════════════════════════════════════════════════

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

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

layout(set = 0, binding = 4, r32ui) uniform readonly uimage2D labelImage;
layout(set = 0, binding = 5, r32f) uniform image2D consumedDepthImage;

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
} pc;

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

vec4 unpackColor(uint packed) {
    float r = float((packed >> 0u)  & 0xFFu) / 255.0;
    float g = float((packed >> 8u)  & 0xFFu) / 255.0;
    float b = float((packed >> 16u) & 0xFFu) / 255.0;
    float a = float((packed >> 24u) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    // Check if this pixel is labeled
    uint label = imageLoad(labelImage, pixel).r;
    if (label == 0u) {
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Find first unconsumed fragment
    uint headIndex = imageLoad(headPointerImage, pixel).r;
    float consumedDepth = imageLoad(consumedDepthImage, pixel).r;

    uint idx = headIndex;
    int steps = 0;

    while (idx != PPLL_NULL && steps < MAX_WALK) {
        float d = nodes[idx].depth;

        if (d > consumedDepth) {
            // Output this fragment's color
            outColor = unpackColor(nodes[idx].color);

            // Update consumed depth — this fragment is now consumed
            imageStore(consumedDepthImage, pixel, vec4(d, 0.0, 0.0, 0.0));
            return;
        }

        idx = nodes[idx].next;
        steps++;
    }

    // Labeled but no unconsumed fragment found (shouldn't happen normally)
    outColor = vec4(0.0, 0.0, 0.0, 0.0);
}