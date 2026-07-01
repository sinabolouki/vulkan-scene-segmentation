#version 450

// ═══════════════════════════════════════════════════════════════════
// GROW PASS (called K times per layer)
//
// For each unlabeled pixel:
//   Check 4-connected neighbors.
//   If a neighbor is labeled AND my first unconsumed fragment is similar
//   (in depth and color) to the neighbor's first unconsumed fragment:
//     → Label this pixel.
//
// "Similar" means:
//   |my.depth - neighbor.depth| < depthThreshold
//   AND euclidean(my.color, neighbor.color) < colorThreshold
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

layout(set = 0, binding = 4, r32ui) uniform uimage2D labelImage;
layout(set = 0, binding = 5, r32f) uniform readonly image2D consumedDepthImage;

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
} pc;

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

// Unpack RGBA8 from uint → vec3 RGB
vec3 unpackRGB(uint packed) {
    float r = float((packed >> 0u)  & 0xFFu) / 255.0;
    float g = float((packed >> 8u)  & 0xFFu) / 255.0;
    float b = float((packed >> 16u) & 0xFFu) / 255.0;
    return vec3(r, g, b);
}

// Find first unconsumed fragment at a pixel
// Returns node index, or PPLL_NULL if none
uint findFirstUnconsumed(ivec2 p) {
    uint idx = imageLoad(headPointerImage, p).r;
    if (idx == PPLL_NULL) return PPLL_NULL;

    float consumed = imageLoad(consumedDepthImage, p).r;
    int steps = 0;

    while (idx != PPLL_NULL && steps < MAX_WALK) {
        if (nodes[idx].depth > consumed) {
            return idx;
        }
        idx = nodes[idx].next;
        steps++;
    }
    return PPLL_NULL;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 imgSize = imageSize(headPointerImage);

    // Already labeled → skip
    uint myLabel = imageLoad(labelImage, pixel).r;
    if (myLabel != 0u) return;

    // Find my first unconsumed fragment
    uint myIdx = findFirstUnconsumed(pixel);
    if (myIdx == PPLL_NULL) return;

    float myDepth = nodes[myIdx].depth;
    vec3  myColor = unpackRGB(nodes[myIdx].color);

    // Check 4-connected neighbors
    ivec2 offsets[4] = ivec2[](
        ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1)
    );

    for (int i = 0; i < 4; i++) {
        ivec2 neighbor = pixel + offsets[i];

        // Bounds check
        if (neighbor.x < 0 || neighbor.y < 0 ||
            neighbor.x >= imgSize.x || neighbor.y >= imgSize.y) continue;

        // Neighbor must be labeled
        uint neighborLabel = imageLoad(labelImage, neighbor).r;
        if (neighborLabel == 0u) continue;

        // Find neighbor's first unconsumed fragment
        uint neighborIdx = findFirstUnconsumed(neighbor);
        if (neighborIdx == PPLL_NULL) continue;

        float neighborDepth = nodes[neighborIdx].depth;
        vec3  neighborColor = unpackRGB(nodes[neighborIdx].color);

        // Depth similarity check
        bool depthOk = abs(myDepth - neighborDepth) <= pc.depthThreshold;

        // Color similarity check (Euclidean distance in RGB)
        float colorDist = length(myColor - neighborColor);
        bool colorOk = colorDist <= pc.colorThreshold;

        if (depthOk && colorOk) {
            imageStore(labelImage, pixel, uvec4(1u, 0, 0, 0));
            return;
        }
    }
}