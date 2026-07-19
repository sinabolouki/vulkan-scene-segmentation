#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Compress Pass (Path Compression)
//
// After the merge pass, the parent buffer forms a forest of trees.
// This pass flattens each tree so that every node points directly to its root.
// After this pass: parent[i] == root of the connected component containing i.
//
// Uses "inline compression" from Bolelli et al. — updates intermediate nodes
// during traversal so concurrent reads benefit immediately.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX      0xFFFFFFFF
#define MAX_FRAGS     32
#define MAX_FIND_ITER 64

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

layout(set = 0, binding = 3, std430) coherent buffer ParentBuffer {
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

// ─── Inline Compress ─────────────────────────────────────────────────────────
// Finds root and simultaneously flattens the path: every node along the way
// gets pointed directly to the root.

uint inlineCompress(uint x) {
    uint startNode = x;

    // Phase 1: find root
    uint root = x;
    for (uint i = 0; i < MAX_FIND_ITER; i++) {
        uint p = parent[root];
        if (p == root) break;
        root = p;
    }

    // Phase 2: compress path — point every node from start to root directly
    uint current = startNode;
    for (uint i = 0; i < MAX_FIND_ITER; i++) {
        uint p = parent[current];
        if (p == root) break;
        parent[current] = root;
        current = p;
    }

    return root;
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        // Compress: parent[idx] = root of its tree
        inlineCompress(idx);
        idx = nodes[idx].next;
        count++;
    }
}