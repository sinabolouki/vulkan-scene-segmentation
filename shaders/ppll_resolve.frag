#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// ═══ PPLL Data Structures (same bindings as collect) ═══

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

// Screen dimensions via push constant
layout(push_constant) uniform PushConstants {
    uint screenWidth;
    uint screenHeight;
} pc;

// Maximum fragments we can sort per pixel
#define MAX_FRAGMENTS 32

// ═══ Unpack color from uint32 ═══
vec4 unpackColor(uint packed) {
    float r = float((packed >> 0u) & 0xFFu) / 255.0;
    float g = float((packed >> 8u) & 0xFFu) / 255.0;
    float b = float((packed >> 16u) & 0xFFu) / 255.0;
    float a = float((packed >> 24u) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);

    // ─── Read the head pointer for this pixel ───
    uint nodeIndex = imageLoad(headPointerImage, pixelCoord).r;

    // If empty pixel, output background
    if (nodeIndex == 0xFFFFFFFFu) {
        outColor = vec4(0.02, 0.02, 0.08, 1.0);
        return;
    }

    // ─── Walk the linked list, collect fragments ───
    float depths[MAX_FRAGMENTS];
    vec4  colors[MAX_FRAGMENTS];
    uint  objectIDs[MAX_FRAGMENTS];
    int   fragCount = 0;

    while (nodeIndex != 0xFFFFFFFFu && fragCount < MAX_FRAGMENTS) {
        FragmentNode node = nodes[nodeIndex];

        depths[fragCount] = node.depth;
        colors[fragCount] = unpackColor(node.color);
        objectIDs[fragCount] = node.objectID;
        fragCount++;

        nodeIndex = node.next;
    }

    // ─── Sort fragments by depth (front-to-back: smallest depth first) ───
    // Simple insertion sort (fine for small arrays on GPU)
    for (int i = 1; i < fragCount; i++) {
        float keyDepth = depths[i];
        vec4  keyColor = colors[i];
        uint  keyObjID = objectIDs[i];

        int j = i - 1;
        while (j >= 0 && depths[j] > keyDepth) {
            depths[j + 1] = depths[j];
            colors[j + 1] = colors[j];
            objectIDs[j + 1] = objectIDs[j];
            j--;
        }
        depths[j + 1] = keyDepth;
        colors[j + 1] = keyColor;
        objectIDs[j + 1] = keyObjID;
    }

    // ─── Composite: back-to-front blending ───
    // Start with background
    vec4 result = vec4(0.02, 0.02, 0.08, 1.0);

    // Blend from back to front (last element = farthest)
    for (int i = fragCount - 1; i >= 0; i--) {
        vec4 src = colors[i];
        // Standard over operator: result = src * src.a + result * (1 - src.a)
        result.rgb = src.rgb * src.a + result.rgb * (1.0 - src.a);
        result.a = 1.0;
    }

    outColor = result;
}