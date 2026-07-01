#version 450

// FIND GLOBAL MIN DEPTH PASS
// For each pixel, walks the sorted linked list and finds the first
// fragment with depth > consumedDepth. That's the local minimum
// (because the list is sorted front-to-back).
// Writes floatBitsToUint(localMin) via atomicMin to a global buffer.
// After this pass, the buffer contains the global minimum depth
// among all unconsumed fragments across the entire screen.
// No color output.

layout(location = 0) in vec2 fragUV;

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

// State buffer:
//   [0] = globalMinUint  (atomicMin target)
//   [1] = consumedDepth as uint bits
layout(set = 0, binding = 3, std430) buffer StateBuffer {
    uint globalMinUint;
    uint consumedDepthBits;
};

#define PPLL_NULL 0xFFFFFFFFu
#define MAX_WALK  32

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);

    uint headIndex = imageLoad(headPointerImage, pixelCoord).r;

    // Empty pixel
    if (headIndex == PPLL_NULL) {
        return;
    }

    // Read consumed depth from state buffer
    float consumedDepth = uintBitsToFloat(consumedDepthBits);

    // Walk sorted list — find first node with depth > consumedDepth
    uint currentIndex = headIndex;
    int steps = 0;

    while (currentIndex != PPLL_NULL && steps < MAX_WALK) {
        float d = nodes[currentIndex].depth;

        if (d > consumedDepth) {
            // Found the local minimum for this pixel
            // Write to global min via atomicMin
            // (positive floats: uint bit ordering == float ordering)
            atomicMin(globalMinUint, floatBitsToUint(d));
            return;
        }

        currentIndex = nodes[currentIndex].next;
        steps++;
    }

    // No unconsumed fragments at this pixel
}