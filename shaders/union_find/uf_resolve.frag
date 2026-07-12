#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Resolve Pass — Outputs front-layer color AND advances head pointer
//
// For each pixel:
//   1. Walk sorted list from head
//   2. Output the first fragment whose root has maxPosition == 0
//   3. Advance head pointer past ALL front-layer fragments
//   4. Next peel starts from the new head
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX         0xFFFFFFFF
#define MAX_FRAGS        32
#define MAX_GLOBAL_ROOTS 32

// NOTE: headPointerImage needs BOTH read and write
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
    uint rootTable[];  // [0..127] = rootIDs, [128..255] = maxPositions
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

bool isRootInFront(uint root) {
    for (uint i = 0; i < MAX_GLOBAL_ROOTS; i++) {
        if (rootTable[i] == NULL_IDX) break;  // early exit
        if (rootTable[i] == root) {
            return rootTable[MAX_GLOBAL_ROOTS + i] == 0u;
        }
    }
    return false;
}


// ─── Normal Resolve + Advance ────────────────────────────────────────────────

vec4 resolveAndAdvance(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    bool rendered = false;
    vec4 result = vec4(0.0);

    if (idx == NULL_IDX) {
        discard;
    }
    uint root = parent[idx];

    if (isRootInFront(root)) {
        // This fragment is part of the current peel's front layer
        if (!rendered) {
            result = unpackColor(nodes[idx].color);
            rendered = true;
        }
        // Continue past it
        idx = nodes[idx].next;
    } else {
        discard;  // This fragment is not part of the front layer, stop here
    }
    
    // Advance head pointer: idx is now the first remaining fragment (or NULL_IDX)
    imageStore(headPointerImage, coord, uvec4(idx));

    return result;
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0); return;
    }

    // Normal resolve: output front layer + advance head
    outColor = resolveAndAdvance(coord);
}