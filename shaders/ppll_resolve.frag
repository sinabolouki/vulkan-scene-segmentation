#version 450

// RESOLVE PASS (per layer)
//
// Walks each pixel's sorted linked list.
// Skips fragments with depth <= consumedDepth (already peeled).
// Collects fragments within [consumedDepth, globalMin + threshold].
// Composites them front-to-back (list is pre-sorted).
// Stops early when depth exceeds the range.
//
// Outputs the layer color for this iteration.

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// PPLL data structures
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

// State buffer
layout(set = 0, binding = 3, std430) readonly buffer StateBuffer {
    uint globalMinUint;
    uint consumedDepthBits;
};

// Push constant: depth threshold for grouping
layout(push_constant) uniform PushConstants {
    float depthThreshold;
} pc;

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

// Unpack RGBA8 from uint
vec4 unpackColor(uint packed) {
    float r = float((packed >> 0u)  & 0xFFu) / 255.0;
    float g = float((packed >> 8u)  & 0xFFu) / 255.0;
    float b = float((packed >> 16u) & 0xFFu) / 255.0;
    float a = float((packed >> 24u) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);

    uint headIndex = imageLoad(headPointerImage, pixelCoord).r;

    // Empty pixel — transparent output
    if (headIndex == PPLL_NULL) {
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Read state
    float globalMinDepth = uintBitsToFloat(globalMinUint);
    float consumedDepth = uintBitsToFloat(consumedDepthBits);
    float maxDepth = globalMinDepth + pc.depthThreshold;

    // Walk sorted list
    uint currentIndex = headIndex;
    int steps = 0;

    // Front-to-back compositing (pre-multiplied alpha)
    vec3  accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    while (currentIndex != PPLL_NULL && steps < MAX_WALK) {
        float d = nodes[currentIndex].depth;

        // Skip already consumed fragments
        if (d <= consumedDepth) {
            currentIndex = nodes[currentIndex].next;
            steps++;
            continue;
        }

        // Early exit: past this layer's depth range (list is sorted)
        if (d > maxDepth) {
            break;
        }

        // This fragment belongs to the current layer
        vec4 fragColor = unpackColor(nodes[currentIndex].color);

        // Front-to-back blending (under operator):
        // accumColor += (1 - accumAlpha) * fragColor.rgb * fragColor.a
        // accumAlpha += (1 - accumAlpha) * fragColor.a
        float oneMinusAlpha = 1.0 - accumAlpha;
        accumColor += oneMinusAlpha * fragColor.rgb * fragColor.a;
        accumAlpha += oneMinusAlpha * fragColor.a;

        // Fully opaque — no point continuing
        if (accumAlpha >= 0.99) {
            break;
        }

        currentIndex = nodes[currentIndex].next;
        steps++;
    }

    outColor = vec4(accumColor, accumAlpha);
}