#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Resolve Pass — Global Layer Assignment by Max Depth
//
// Layers are assigned based on each root's maximum depth:
//   Layer 0 = roots with smallest maxDepth (closest surfaces)
//   Layer 1 = roots with next maxDepth (deeper surfaces)
//   ...
//
// Two roots with the SAME maxDepth get the SAME layer rank.
// → If they don't overlap at any pixel, both show correctly.
// → If they do overlap, the shallower fragment at that pixel wins.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX         0xFFFFFFFF
#define MAX_FRAGS        32
#define MAX_GLOBAL_ROOTS 128

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) buffer ParentBuffer {
    uint parent[];
};

layout(set = 0, binding = 4, std430) buffer RootTable {
    uint rootTable[];  // [i*2] = rootID, [i*2+1] = maxDepth (as uint bits)
};

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
};

layout(location = 0) out vec4 outColor;

// ─── Helpers ─────────────────────────────────────────────────────────────────

vec4 unpackColor(uint packed) {
    return vec4(
        float(packed & 0xFFu) / 255.0,
        float((packed >> 8u) & 0xFFu) / 255.0,
        float((packed >> 16u) & 0xFFu) / 255.0,
        float((packed >> 24u) & 0xFFu) / 255.0
    );
}

vec3 hashColor(uint id) {
    uint h = id;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = (h >> 16u) ^ h;
    return vec3(
        float(h & 0xFFu) / 255.0,
        float((h >> 8u) & 0xFFu) / 255.0,
        float((h >> 16u) & 0xFFu) / 255.0
    );
}

// ─── Get Layer Rank for a Root ───────────────────────────────────────────────
// Rank = number of OTHER registered roots that have a STRICTLY smaller maxDepth.
// Roots with the same maxDepth get the SAME rank (can share a layer).

// Layout: rootTable[0..7] = rootIDs, rootTable[8..15] = maxDepths

uint getLayerRank(uint root) {
    // Find this root's maxDepth in the table
    uint myMaxDepth = NULL_IDX;
    for (uint i = 0; i < MAX_GLOBAL_ROOTS; i++) {
        if (rootTable[i] == root) {
            myMaxDepth = rootTable[MAX_GLOBAL_ROOTS + i];
            break;
        }
    }
    if (myMaxDepth == NULL_IDX) return NULL_IDX;  // root not registered

    return myMaxDepth;
}

// ─── Debug Modes ─────────────────────────────────────────────────────────────

vec4 debugRootColor(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);
    uint root = parent[idx];
    return vec4(hashColor(root), 1.0);
}

vec4 debugRootCount(ivec2 coord) {
    uint seenRoots[MAX_FRAGS];
    uint numSeen = 0;
    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;
    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];
        bool seen = false;
        for (uint i = 0; i < numSeen; i++) {
            if (seenRoots[i] == root) { seen = true; break; }
        }
        if (!seen && numSeen < MAX_FRAGS) { seenRoots[numSeen] = root; numSeen++; }
        idx = nodes[idx].next; count++;
    }
    if (numSeen == 0) return vec4(0.0);
    if (numSeen == 1) return vec4(0.0, 0.8, 0.0, 1.0);
    if (numSeen == 2) return vec4(0.9, 0.9, 0.0, 1.0);
    if (numSeen == 3) return vec4(1.0, 0.5, 0.0, 1.0);
    return vec4(1.0, 0.0, 0.0, 1.0);
}

vec4 debugDepth(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);
    float v = 1.0 - nodes[idx].depth;
    return vec4(v, v, v, 1.0);
}

vec4 debugFragCount(ivec2 coord) {
    uint count = 0;
    uint idx = imageLoad(headPointerImage, coord).r;
    while (idx != NULL_IDX && count < MAX_FRAGS) { count++; idx = nodes[idx].next; }
    if (count == 0) return vec4(0.0);
    if (count == 1) return vec4(0.0, 0.0, 0.5, 1.0);
    if (count == 2) return vec4(0.0, 0.5, 1.0, 1.0);
    if (count == 3) return vec4(0.0, 1.0, 0.5, 1.0);
    if (count == 4) return vec4(1.0, 1.0, 0.0, 1.0);
    return vec4(1.0, 0.0, 0.0, 1.0);
}

vec4 debugBypass(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);
    return unpackColor(nodes[idx].color);
}

// Mode 105: visualize global layer rank of front fragment
vec4 debugGlobalRank(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);
    uint root = parent[idx];
    uint rank = getLayerRank(root);
    if (rank == 0u) return vec4(1.0, 0.0, 0.0, 1.0);  // red
    if (rank == 1u) return vec4(0.0, 1.0, 0.0, 1.0);  // green
    if (rank == 2u) return vec4(0.0, 0.0, 1.0, 1.0);  // blue
    if (rank == 3u) return vec4(1.0, 1.0, 0.0, 1.0);  // yellow
    return vec4(0.5, 0.5, 0.5, 1.0);
}

// ─── Normal Resolve ──────────────────────────────────────────────────────────
// Walk the sorted list. Output the FIRST fragment whose root's global rank
// matches the target layer.

vec4 normalResolve(ivec2 coord, uint layer) {
    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];
        uint rank = getLayerRank(root);

        if (rank == layer) {
            return unpackColor(nodes[idx].color);
        }

        idx = nodes[idx].next;
        count++;
    }

    return vec4(0.0);  // transparent — this surface doesn't exist at this pixel
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0); return;
    }

    if (targetLayer == 100u) { outColor = debugRootColor(coord); return; }
    if (targetLayer == 101u) { outColor = debugRootCount(coord); return; }
    if (targetLayer == 102u) { outColor = debugDepth(coord); return; }
    if (targetLayer == 103u) { outColor = debugFragCount(coord); return; }
    if (targetLayer == 104u) { outColor = debugBypass(coord); return; }
    if (targetLayer == 105u) { outColor = debugGlobalRank(coord); return; }

    outColor = normalResolve(coord, targetLayer);
}