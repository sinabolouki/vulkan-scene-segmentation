#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// ═══ PPLL Data Structures ═══

layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint color;
    float depth;
    uint objectID;
    uint next;
};

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) readonly buffer AtomicCounter {
    uint counter;
};

layout(push_constant) uniform PushConstants {
    uint screenWidth;
    uint screenHeight;
} pc;

#define MAX_FRAGMENTS 32

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);

    // Read head pointer
    uint nodeIndex = imageLoad(headPointerImage, pixelCoord).r;

    // Count fragments in this pixel's list
    int fragCount = 0;
    while (nodeIndex != 0xFFFFFFFFu && fragCount < MAX_FRAGMENTS) {
        nodeIndex = nodes[nodeIndex].next;
        fragCount++;
    }

    // ═══ Heatmap Visualization ═══
    // 0 fragments:  black (background)
    // 1 fragment:   dark blue
    // 2-3:          green
    // 4-6:          yellow
    // 7+:           red/white (high overdraw)

    if (fragCount == 0) {
        outColor = vec4(0.02, 0.02, 0.08, 1.0); // Background
        return;
    }

    // Normalize to [0, 1] range (8 = expected average)
    float t = clamp(float(fragCount) / 8.0, 0.0, 1.0);

    // Color ramp: blue → green → yellow → red
    vec3 color;
    if (t < 0.33) {
        // Blue to Green
        float s = t / 0.33;
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), s);
    } else if (t < 0.66) {
        // Green to Yellow
        float s = (t - 0.33) / 0.33;
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), s);
    } else {
        // Yellow to Red
        float s = (t - 0.66) / 0.34;
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), s);
    }

    outColor = vec4(color, 1.0);
}