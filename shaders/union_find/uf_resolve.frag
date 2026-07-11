#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Resolve Pass (with Debug Modes)
//
// targetLayer meanings:
//   0-3: Normal mode — output N-th distinct surface color
//   100: DEBUG — visualize front fragment's ROOT as a hashed color
//   101: DEBUG — visualize NUMBER of distinct roots per pixel
//   102: DEBUG — visualize front fragment's DEPTH as grayscale
//   103: DEBUG — show which pixels have >1 fragment (overlap regions)
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX  0xFFFFFFFF
#define MAX_FRAGS 32
#define MAX_ROOTS 8

// ─── Descriptors ─────────────────────────────────────────────────────────────

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

// ─── Push Constants ──────────────────────────────────────────────────────────

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
};

// ─── Output ──────────────────────────────────────────────────────────────────

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

// Hash a uint to a visually distinct RGB color
vec3 hashColor(uint id) {
    // Improved hash: spread bits to avoid similar colors for adjacent IDs
    uint h = id;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = (h >> 16u) ^ h;

    return vec3(
        float((h)        & 0xFFu) / 255.0,
        float((h >> 8u)  & 0xFFu) / 255.0,
        float((h >> 16u) & 0xFFu) / 255.0
    );
}

// ─── Debug Mode: Root Visualization ──────────────────────────────────────────

vec4 debugRootColor(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);

    uint root = parent[idx];
    return vec4(hashColor(root), 1.0);
}

// ─── Debug Mode: Distinct Root Count ─────────────────────────────────────────

vec4 debugRootCount(ivec2 coord) {
    uint seenRoots[MAX_ROOTS];
    uint numSeen = 0;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];
        bool seen = false;
        for (uint i = 0; i < numSeen; i++) {
            if (seenRoots[i] == root) { seen = true; break; }
        }
        if (!seen && numSeen < MAX_ROOTS) {
            seenRoots[numSeen] = root;
            numSeen++;
        }
        idx = nodes[idx].next;
        count++;
    }

    // Color by count: 1=green, 2=yellow, 3=orange, 4+=red
    if (numSeen == 0) return vec4(0.0);
    if (numSeen == 1) return vec4(0.0, 0.8, 0.0, 1.0);
    if (numSeen == 2) return vec4(0.9, 0.9, 0.0, 1.0);
    if (numSeen == 3) return vec4(1.0, 0.5, 0.0, 1.0);
    return vec4(1.0, 0.0, 0.0, 1.0);
}

// ─── Debug Mode: Depth Visualization ─────────────────────────────────────────

vec4 debugDepth(ivec2 coord) {
    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) return vec4(0.0);

    float d = nodes[idx].depth;
    // Map depth [0,1] to grayscale (invert so near=bright)
    float v = 1.0 - d;
    return vec4(v, v, v, 1.0);
}

// ─── Debug Mode: Fragment Count (overlap) ────────────────────────────────────

vec4 debugFragCount(ivec2 coord) {
    uint count = 0;
    uint idx = imageLoad(headPointerImage, coord).r;
    while (idx != NULL_IDX && count < MAX_FRAGS) {
        count++;
        idx = nodes[idx].next;
    }

    if (count == 0) return vec4(0.0);
    if (count == 1) return vec4(0.0, 0.0, 0.5, 1.0);  // dark blue
    if (count == 2) return vec4(0.0, 0.5, 1.0, 1.0);  // light blue
    if (count == 3) return vec4(0.0, 1.0, 0.5, 1.0);  // cyan-green
    if (count == 4) return vec4(1.0, 1.0, 0.0, 1.0);  // yellow
    return vec4(1.0, 0.0, 0.0, 1.0);                   // red = 5+
}

// ─── Normal Mode: Layer Resolve ──────────────────────────────────────────────

vec4 normalResolve(ivec2 coord, uint layer) {
    uint seenRoots[MAX_ROOTS];
    uint numSeenRoots = 0;
    uint currentLayer = 0;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];

        bool alreadySeen = false;
        for (uint i = 0; i < numSeenRoots; i++) {
            if (seenRoots[i] == root) {
                alreadySeen = true;
                break;
            }
        }

        if (!alreadySeen) {
            if (currentLayer == layer) {
                return unpackColor(nodes[idx].color);
            }
            if (numSeenRoots < MAX_ROOTS) {
                seenRoots[numSeenRoots] = root;
                numSeenRoots++;
            }
            currentLayer++;
        }

        idx = nodes[idx].next;
        count++;
    }

    return vec4(0.0);
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0);
        return;
    }

    // Debug modes
    if (targetLayer == 100u) {
        outColor = debugRootColor(coord);
        return;
    }
    if (targetLayer == 101u) {
        outColor = debugRootCount(coord);
        return;
    }
    if (targetLayer == 102u) {
        outColor = debugDepth(coord);
        return;
    }
    if (targetLayer == 103u) {
        outColor = debugFragCount(coord);
        return;
    }

    // Normal resolve
    outColor = normalResolve(coord, targetLayer);
}