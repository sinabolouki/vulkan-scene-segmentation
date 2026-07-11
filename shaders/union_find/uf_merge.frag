#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Union-Find Merge Pass
//
// For each pixel: collect all local fragments, then for each of 8 neighbors
// walk their entire linked list. If any pair (local, neighbor) is similar
// in depth and color, Union() their parent pointers.
// Also unions within the same pixel for fragments at similar depth+color.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX     0xFFFFFFFF
#define MAX_FRAGS    32
#define MAX_FIND_ITER 64

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
    uint  targetLayer;  // unused in this pass
};

// ─── Helper: Unpack RGBA8 ────────────────────────────────────────────────────

vec3 unpackColor(uint packed) {
    return vec3(
        float(packed & 0xFFu) / 255.0,
        float((packed >> 8u) & 0xFFu) / 255.0,
        float((packed >> 16u) & 0xFFu) / 255.0
    );
}

// ─── Helper: Similarity Test ─────────────────────────────────────────────────

bool isSimilar(uint idxA, uint idxB) {
    // Depth check (cheap, do first)
    float dA = nodes[idxA].depth;
    float dB = nodes[idxB].depth;
    if (abs(dA - dB) > depthThreshold) return false;

    // Color check
    vec3 cA = unpackColor(nodes[idxA].color);
    vec3 cB = unpackColor(nodes[idxB].color);
    float dist = distance(cA, cB);
    if (dist > colorThreshold) return false;

    return true;
}

// ─── Union-Find: Find Root ───────────────────────────────────────────────────
// Follows parent pointers until reaching a root (parent[x] == x).
// Bounded to MAX_FIND_ITER to prevent infinite loops from corruption.

uint find(uint x) {
    for (uint i = 0; i < MAX_FIND_ITER; i++) {
        uint p = parent[x];
        if (p == x) break;
        x = p;
    }
    return x;
}

// ─── Union-Find: Union ───────────────────────────────────────────────────────
// Merges the trees containing nodes a and b.
// Uses atomicMin to deterministically resolve races:
//   - Always points higher root → lower root
//   - If atomicMin fails (someone else changed it), retry with updated value
//
// From: Bolelli et al., Algorithm 1 (TPDS 2026)

void uf_union(uint a, uint b) {
    bool done = false;
    for (uint iter = 0; iter < MAX_FIND_ITER && !done; iter++) {
        a = find(a);
        b = find(b);

        if (a < b) {
            uint old = atomicMin(parent[b], a);
            done = (old == b);   // success: parent[b] was b, now is a
            b = old;             // if failed, retry from old value
        } else if (b < a) {
            uint old = atomicMin(parent[a], b);
            done = (old == a);
            a = old;
        } else {
            done = true;         // already in same component
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Bounds check (shouldn't be needed with correct viewport, but safety)
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    // ─── Collect this pixel's fragments ──────────────────────────────────────
    uint myFrags[MAX_FRAGS];
    uint myCount = 0;

    uint idx = imageLoad(headPointerImage, coord).r;
    while (idx != NULL_IDX && myCount < MAX_FRAGS) {
        myFrags[myCount] = idx;
        myCount++;
        idx = nodes[idx].next;
    }

    if (myCount == 0) return;  // empty pixel, nothing to do

    // ─── Union within same pixel ─────────────────────────────────────────────
    // Fragments in the same pixel at similar depth+color likely belong to the
    // same surface (e.g., triangle seams of the same mesh).
    for (uint i = 0; i < myCount; i++) {
        for (uint j = i + 1; j < myCount; j++) {
            if (isSimilar(myFrags[i], myFrags[j])) {
                uf_union(myFrags[i], myFrags[j]);
            }
        }
    }

    // ─── Union with 8 neighbors ──────────────────────────────────────────────
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // skip self

            ivec2 ncoord = coord + ivec2(dx, dy);

            // Bounds check
            if (ncoord.x < 0 || ncoord.x >= int(screenWidth) ||
                ncoord.y < 0 || ncoord.y >= int(screenHeight)) continue;

            // Walk neighbor's entire linked list
            uint nIdx = imageLoad(headPointerImage, ncoord).r;
            uint nCount = 0;

            while (nIdx != NULL_IDX && nCount < MAX_FRAGS) {
                // Compare this neighbor fragment with ALL my fragments
                for (uint i = 0; i < myCount; i++) {
                    if (isSimilar(myFrags[i], nIdx)) {
                        uf_union(myFrags[i], nIdx);
                    }
                }
                nIdx = nodes[nIdx].next;
                nCount++;
            }
        }
    }
}