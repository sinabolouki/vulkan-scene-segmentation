#version 450

// PER-PIXEL SORT PASS
// Walks each pixel's linked list, sorts fragments by depth
// (front-to-back), and rewrites the `next` pointers in the SSBO
// so subsequent passes can do simple linear traversal.
// Runs ONCE after the collect pass.
// No color output — only modifies the node buffer and head image.

layout(location = 0) in vec2 fragUV;

// PPLL data structures
layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

#define MAX_FRAGMENTS 32
#define PPLL_NULL 0xFFFFFFFFu

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);

    // Read head pointer for this pixel
    uint headIndex = imageLoad(headPointerImage, pixelCoord).r;

    // Empty pixel — nothing to sort
    if (headIndex == PPLL_NULL) {
        return;
    }

    // ─── Collect all node indices and depths into local arrays ───
    uint  nodeIndices[MAX_FRAGMENTS];
    float nodeDepths[MAX_FRAGMENTS];
    int   count = 0;

    uint currentIndex = headIndex;
    while (currentIndex != PPLL_NULL && count < MAX_FRAGMENTS) {
        nodeIndices[count] = currentIndex;
        nodeDepths[count] = nodes[currentIndex].depth;
        count++;
        currentIndex = nodes[currentIndex].next;
    }

    // Single fragment — already sorted
    if (count <= 1) {
        return;
    }

    // ─── Insertion sort by depth (ascending = front-to-back) ───
    for (int i = 1; i < count; i++) {
        float keyDepth = nodeDepths[i];
        uint  keyIndex = nodeIndices[i];

        int j = i - 1;
        while (j >= 0 && nodeDepths[j] > keyDepth) {
            nodeDepths[j + 1] = nodeDepths[j];
            nodeIndices[j + 1] = nodeIndices[j];
            j--;
        }
        nodeDepths[j + 1] = keyDepth;
        nodeIndices[j + 1] = keyIndex;
    }

    // ─── Rewrite linked list pointers in sorted order ───
    // Update head to point to the frontmost fragment
    imageStore(headPointerImage, pixelCoord, uvec4(nodeIndices[0], 0, 0, 0));

    // Chain nodes in sorted order
    for (int i = 0; i < count - 1; i++) {
        nodes[nodeIndices[i]].next = nodeIndices[i + 1];
    }

    // Last node points to null
    nodes[nodeIndices[count - 1]].next = PPLL_NULL;
}