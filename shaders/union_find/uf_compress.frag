#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Compress Pass (Path Compression)
//
// After the merge pass, the parent buffer forms a forest of trees.
// This pass flattens each tree so that every node points directly to its root.
// After this pass: parent[i] == root of the connected component containing i.
//
// Parent reads and updates are atomic because paths from different starting
// nodes can overlap. Merging must finish, with a host memory dependency,
// before compression begins. The root of each component is then stable.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX      0xFFFFFFFFu
#define MAX_FRAGS     32u

// ─── Descriptors ─────────────────────────────────────────────────────────────

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    uint  unshadedColor;  // Packed RGBA8 (unlit base color)
    float depth;
    float normalX;     // View-space normal X
    float normalY;     // View-space normal Y
    float normalZ;     // View-space normal Z
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) readonly buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) coherent buffer ParentBuffer {
    uint parent[];
};

// ─── Push Constants ──────────────────────────────────────────────────────────

layout(push_constant) uniform PushConstants {
    float planeThreshold;
    float proximityScale;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
};

// ─── Inline Compress ─────────────────────────────────────────────────────────
// First find a confirmed root, then flatten the path atomically.
// Neither phase treats a traversal budget as evidence of reaching a root.

uint readParent(uint x) {
    return atomicAdd(parent[x], 0u);
}

uint inlineCompress(uint x) {
    // Phase 1: find root
    uint root = x;
    while (true) {
        uint p = readParent(root);
        if (p == root) break;
        root = p;
    }

    // Phase 2: atomically redirect visited nodes to the stable root.
    // atomicMin returns the old parent, so no separate ordinary read is needed.
    uint current = x;
    while (current != root) {
        current = atomicMin(parent[current], root);
    }

    return root;
}

bool validNodeIndex(uint index, uint validNodeCount) {
    return index != NULL_IDX && index < validNodeCount;
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    // Helper invocations have no usable return values from buffer atomics.
    if (gl_HelperInvocation) return;

    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x < 0 || coord.y < 0 ||
        coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    uint validNodeCount = min(
        counter,
        min(uint(nodes.length()), uint(parent.length()))
    );
    if (validNodeCount == 0u) return;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0u;

    while (validNodeIndex(idx, validNodeCount) && count < MAX_FRAGS) {
        // Compress: parent[idx] = root of its tree
        inlineCompress(idx);
        idx = nodes[idx].next;
        count++;
    }
}
