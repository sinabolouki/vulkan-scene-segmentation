#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Union-Find Merge Pass — Rank-Constrained Soft-Cost Connectivity
//
// Key insight: fragments are only merged across pixel boundaries if they
// appear at COMPATIBLE DEPTH RANKS in their respective sorted lists.
// This ensures that depth-coherent surfaces form separate components,
// even when they belong to the same geometric object (e.g., front and
// back faces of a concave bowl).
//
// For each local fragment at rank k, we only consider neighbor fragments
// at ranks [k - RANK_WINDOW, k + RANK_WINDOW].
//
// The soft-cost function is unchanged — it still checks geometric
// compatibility via tangent-plane residual, normal deviation, and depth
// proximity. The rank constraint is an additional structural filter.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX       0xFFFFFFFFu
#define MAX_FRAGS      32u
#define MAX_FIND_ITER  64u

// How many ranks above/below to consider when matching across pixels.
// ±1 handles single-fragment insertions between pixels.
// ±2 is more robust for complex geometry with variable overdraw.
#define RANK_WINDOW    2u

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;       // Linear depth normalized to [0,1]
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

layout(push_constant) uniform PushConstants {
    float planeThreshold;    // Reference plane-residual scale
    float proximityScale;    // Reference proximity scale
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;       // Unused in merge
    float nearPlane;
    float farPlane;
    float projX;             // projection[0][0]
    float projY;             // projection[1][1]
    float colorThreshold;
} pc;

// ─── Soft-cost weights ───────────────────────────────────────────────────────
const float W_PLANE        = 0.25;
const float W_NORMAL       = 0.25;
const float W_DEPTH        = 0.25;
const float W_COLOR        = 0.25;  // Set > 0 to enable color-based filtering
const float COST_THRESHOLD = 1.0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

vec3 getNodeNormal(uint idx) {
    vec3 n = vec3(nodes[idx].normalX, nodes[idx].normalY, nodes[idx].normalZ);
    float len2 = dot(n, n);
    if (len2 < 1e-12) {
        return vec3(0.0, 0.0, -1.0);
    }
    return n * inversesqrt(len2);
}

vec3 reconstructViewPosition(ivec2 pixel, float normalizedLinearDepth) {
    float viewDistance = pc.nearPlane
        + clamp(normalizedLinearDepth, 0.0, 1.0) * (pc.farPlane - pc.nearPlane);

    vec2 ndc = (vec2(pixel) + 0.5) / vec2(float(pc.screenWidth), float(pc.screenHeight));
    ndc = ndc * 2.0 - 1.0;
    ndc.y = -ndc.y; // Vulkan Y-flip

    return vec3(
        ndc.x * viewDistance / max(abs(pc.projX), 1e-6),
        ndc.y * viewDistance / max(abs(pc.projY), 1e-6),
        -viewDistance
    );
}

float computePixelStep(ivec2 pixelA, ivec2 pixelB, float refDepth) {
    vec3 refA = reconstructViewPosition(pixelA, refDepth);
    vec3 refB = reconstructViewPosition(pixelB, refDepth);
    return length(refB - refA);
}

vec3 unpackColorRGB(uint packed) {
    return vec3(
        float(packed & 0xFFu) / 255.0,
        float((packed >> 8u) & 0xFFu) / 255.0,
        float((packed >> 16u) & 0xFFu) / 255.0
    );
}

float colorDistance(uint packedA, uint packedB) {
    vec3 colorA = unpackColorRGB(packedA);
    vec3 colorB = unpackColorRGB(packedB);

    float lumA = dot(colorA, vec3(0.299, 0.587, 0.114));
    float lumB = dot(colorB, vec3(0.299, 0.587, 0.114));

    if (lumA < 0.02 && lumB < 0.02) return 0.0;
    if (min(lumA, lumB) < 0.02)     return abs(lumA - lumB);

    vec3 chromA = colorA / max(lumA, 0.01);
    vec3 chromB = colorB / max(lumB, 0.01);

    return length(chromA - chromB);
}

// ─── Soft-Cost Connectivity ──────────────────────────────────────────────────

float connectionCost(uint idxA, uint idxB, ivec2 pixelA, ivec2 pixelB) {
    // Optional color veto
    // if (W_COLOR > 0.0) {
    //     float colDist = colorDistance(nodes[idxA].color, nodes[idxB].color);
    //     if (colDist > pc.colorThreshold) return 1e30;
    // }

    float depthA = nodes[idxA].depth;
    float depthB = nodes[idxB].depth;

    vec3 posA = reconstructViewPosition(pixelA, depthA);
    vec3 posB = reconstructViewPosition(pixelB, depthB);
    vec3 delta = posB - posA;

    vec3 normalA = getNodeNormal(idxA);
    vec3 normalB = getNodeNormal(idxB);

    float normalDot = dot(normalA, normalB);

    // Adaptive spatial scale from actual pixel spacing at this depth
    float refDepth = 0.5 * (depthA + depthB);
    float pixelStep = computePixelStep(pixelA, pixelB, refDepth);

    // ── Term 1: normal deviation ─────────────────────────────────────────
    float normalTerm = (1.0 - normalDot) * 0.5;

    // ── Term 2: depth-space proximity ────────────────────────────────────
    float depthDelta = length(delta);
    float depthTerm  = depthDelta / max(pc.proximityScale * pixelStep, 1e-6);

    // ── Term 3: tangent-plane residual ───────────────────────────────────
    float planeTerm;
    if (normalDot > 0.25) {
        float residualA   = abs(dot(normalA,  delta));
        float residualB   = abs(dot(normalB, -delta));
        float maxResidual = max(residualA, residualB);
        planeTerm = maxResidual / max(pc.planeThreshold * pixelStep, 1e-6);
    } else {
        planeTerm = 0.0;
    }

    // ── Term 4: color (optional) ─────────────────────────────────────────
    float colorTerm = 0.0;
    if (W_COLOR > 0.0) {
        float colDist = colorDistance(nodes[idxA].color, nodes[idxB].color);
        colorTerm = colDist / max(pc.colorThreshold, 1e-6);
    }

    float cost = W_PLANE  * planeTerm
               + W_NORMAL * normalTerm
               + W_DEPTH  * depthTerm
               + W_COLOR  * colorTerm;

    if (cost >= COST_THRESHOLD) return 1e30;
    return cost;
}

// ─── Union-Find ──────────────────────────────────────────────────────────────

uint findRoot(uint x) {
    for (uint i = 0u; i < MAX_FIND_ITER; ++i) {
        uint p = parent[x];
        if (p == x) break;
        x = p;
    }
    return x;
}

void unionSets(uint a, uint b) {
    for (uint iteration = 0u; iteration < MAX_FIND_ITER; ++iteration) {
        a = findRoot(a);
        b = findRoot(b);

        if (a == b) return;

        uint high = max(a, b);
        uint low  = min(a, b);

        uint previous = atomicMin(parent[high], low);
        if (previous == high || previous == low) return;
    }
}

bool validNodeIndex(uint index, uint validNodeCount) {
    return index != NULL_IDX && index < validNodeCount;
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coordinate = ivec2(gl_FragCoord.xy);

    if (coordinate.x < 0 || coordinate.y < 0 ||
        coordinate.x >= int(pc.screenWidth) ||
        coordinate.y >= int(pc.screenHeight)) {
        return;
    }

    uint validNodeCount = min(
        counter,
        min(uint(nodes.length()), uint(parent.length()))
    );
    if (validNodeCount == 0u) return;

    // ─── Load local fragments into array (sorted front-to-back) ───────────
    uint localFragments[MAX_FRAGS];
    uint localCount = 0u;

    uint index = imageLoad(headPointerImage, coordinate).r;
    while (validNodeIndex(index, validNodeCount) && localCount < MAX_FRAGS) {
        localFragments[localCount++] = index;
        index = nodes[index].next;
    }

    if (localCount == 0u) return;

    // ─── Symmetric 4-neighborhood: N, W, NW, NE ──────────────────────────
    const ivec2 offsets[4] = ivec2[4](
        ivec2( 0, -1),   // N
        ivec2(-1,  0),   // W
        ivec2(-1, -1),   // NW
        ivec2( 1, -1)    // NE
    );

    for (int dir = 0; dir < 4; ++dir) {
        ivec2 neighborCoordinate = coordinate + offsets[dir];

        if (neighborCoordinate.x < 0 || neighborCoordinate.y < 0 ||
            neighborCoordinate.x >= int(pc.screenWidth) ||
            neighborCoordinate.y >= int(pc.screenHeight)) {
            continue;
        }

        // ─── Pre-load neighbor fragments into array ──────────────────────
        uint neighborFragments[MAX_FRAGS];
        uint neighborCount = 0u;

        uint neighborIndex = imageLoad(headPointerImage, neighborCoordinate).r;
        while (validNodeIndex(neighborIndex, validNodeCount) && neighborCount < MAX_FRAGS) {
            neighborFragments[neighborCount++] = neighborIndex;
            neighborIndex = nodes[neighborIndex].next;
        }

        if (neighborCount == 0u) continue;

        // ─── Rank-constrained matching ───────────────────────────────────
        // For each local fragment at rank k, only consider neighbor
        // fragments at ranks [k - RANK_WINDOW, k + RANK_WINDOW].
        // This prevents merging across depth discontinuities within
        // the same pixel column (e.g., bowl front ↔ bowl back).

        for (uint localRank = 0u; localRank < localCount; ++localRank) {

            uint  bestNeighbor = NULL_IDX;
            float bestCost     = COST_THRESHOLD;

            // Compute rank search bounds (clamped to valid range)
            uint rankMin = (localRank > RANK_WINDOW) ? (localRank - RANK_WINDOW) : 0u;
            uint rankMax = min(localRank + RANK_WINDOW, neighborCount - 1u);

            for (uint neighborRank = rankMin; neighborRank <= rankMax; ++neighborRank) {

                float cost = connectionCost(
                    localFragments[localRank],
                    neighborFragments[neighborRank],
                    coordinate,
                    neighborCoordinate
                );

                if (cost < bestCost) {
                    bestCost     = cost;
                    bestNeighbor = neighborFragments[neighborRank];
                }
            }

            if (bestNeighbor != NULL_IDX) {
                unionSets(localFragments[localRank], bestNeighbor);
            }
        }
    }
}