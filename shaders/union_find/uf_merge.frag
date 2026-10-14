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
#define RANK_WINDOW    8u

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    uint  unshadedColor;  // Packed RGBA8 (unlit base color)
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
    uint  mergeMode;          // 0 = balanced, 1 = color priority, 2 = space priority
} pc;

// ─── Soft-cost weights ───────────────────────────────────────────────────────
const float W_PLANE        = 0.25;
const float W_NORMAL       = 0.25;
const float W_DEPTH        = 0.25;
const float W_COLOR        = 0.25;  // Set > 0 to enable color-based filtering
const float COST_THRESHOLD = 1.0;
const float VETO_THRESHOLD = 0.7;  // If any term exceeds this, reject immediately
const float ACCEPT_THRESHOLD = 0.0;  // If all terms are below this, accept immediately

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

    float depthA = nodes[idxA].depth;
    float depthB = nodes[idxB].depth;

    float dA = clamp(depthA, 0.0, 1.0);
    float dB = clamp(depthB, 0.0, 1.0);
    float depthRange = pc.farPlane - pc.nearPlane;

    float meanDistance = pc.nearPlane
        + 0.5 * (dA + dB) * depthRange;

    // Subtract normalized depths before converting to view-space units.
    float distanceDifference = (dB - dA) * depthRange;

    vec2 viewport = vec2(pc.screenWidth, pc.screenHeight);
    vec2 inverseProjection = 1.0 / max(
        abs(vec2(pc.projX, pc.projY)),
        vec2(1e-6)
    );

    // Construct the difference between camera rays directly from pixels.
    vec2 rayDifferenceXY = 2.0 * vec2(pixelB - pixelA)
        / viewport * inverseProjection * vec2(1.0, -1.0);

    vec3 rayDifference = vec3(rayDifferenceXY, 0.0);

    vec2 midpointNDC =
        (vec2(pixelA) + vec2(pixelB) + vec2(1.0)) / viewport - 1.0;

    vec3 rayMidpoint = vec3(
        midpointNDC * inverseProjection * vec2(1.0, -1.0),
        -1.0
    );

    vec3 lateralStep = meanDistance * rayDifference;

    // Equivalent displacement, without posB - posA cancellation.
    vec3 delta = lateralStep + distanceDifference * rayMidpoint;
    float pixelStep = length(lateralStep);

    vec3 normalA = getNodeNormal(idxA);
    vec3 normalB = getNodeNormal(idxB);
    float normalDot = clamp(dot(normalA, normalB), -1.0, 1.0);
    float normalTerm = 0.5 * (1.0 - normalDot);

    vec3 normalSum = normalA + normalB;
    vec3 meanNormal = normalSum * inversesqrt(
        max(dot(normalSum, normalSum), 1e-12)
    );

    float denominator = dot(meanNormal, rayMidpoint);
    float safeDenominator = (denominator < 0.0 ? -1.0 : 1.0)
        * max(abs(denominator), 1e-4);

    // Expected displacement along the local tangent plane.
    vec3 tangentStep = lateralStep - rayMidpoint
        * (dot(meanNormal, lateralStep) / safeDenominator);

    float tangentSpacing = clamp(
        length(tangentStep),
        pixelStep,
        6.0 * pixelStep
    );

    // Gradual adjustment instead of an abrupt normalDot > 0.9 switch.
    float normalAgreement = smoothstep(0.5, 0.95, normalDot);
    float expectedStep = mix(pixelStep, tangentSpacing, normalAgreement);

    float depthTerm =
        length(delta) / max(pc.proximityScale * expectedStep, 1e-6);

    float planeTerm;
    if (normalDot > 0.25) {
        float residualA   = abs(dot(normalA,  delta));
        float residualB   = abs(dot(normalB, -delta));
        float maxResidual = max(residualA, residualB);
        planeTerm = maxResidual / max(pc.planeThreshold * pixelStep, 1e-6);
    } else {
        planeTerm = 0.0;
    }

    float spatialCost = W_PLANE * planeTerm + W_NORMAL * normalTerm + W_DEPTH * depthTerm;
    // Normalize so that 1.0 = threshold (spatial terms sum to 0.75 max weight)
    float spatialNorm = spatialCost / (W_PLANE + W_NORMAL + W_DEPTH);

    // ── Color term ───────────────────────────────────────────────────────
    float colDist = colorDistance(nodes[idxA].unshadedColor, nodes[idxB].unshadedColor);
    float colorNorm = colDist / max(pc.colorThreshold, 1e-6);

    // ── Mode 0: Balanced (weighted sum of all terms) ─────────────────────
    if (pc.mergeMode == 0u) {
        float cost = W_PLANE  * planeTerm
                   + W_NORMAL * normalTerm
                   + W_DEPTH  * depthTerm
                   + W_COLOR  * colorNorm;

        return (cost >= COST_THRESHOLD) ? 1e30 : cost;
    }

    // ── Mode 1: Color priority ───────────────────────────────────────────
    //    If colors differ beyond threshold → reject immediately.
    //    Otherwise, decide using spatial terms only.
    if (pc.mergeMode == 1u) {
        if (colorNorm < ACCEPT_THRESHOLD) return colorNorm * 0.5;  // Accept early
        if (colorNorm >= VETO_THRESHOLD)  return 1e30;             // Reject early
        return (spatialNorm >= COST_THRESHOLD) ? 1e30 : spatialNorm;            // Spatial decides
    }

    // ── Mode 2: Space priority ───────────────────────────────────────────
    //    Space very similar  → accept immediately (skip color)
    //    Space very different → reject immediately (skip color)
    //    Space in between    → color decides
    if (pc.mergeMode == 2u) {
        if (spatialNorm < ACCEPT_THRESHOLD) return spatialNorm * 0.5;  // Accept early
        if (spatialNorm >= VETO_THRESHOLD)  return 1e30;               // Reject early
        return (colorNorm >= COST_THRESHOLD) ? 1e30 : colorNorm;                    // Color decides
    }

    // Fallback (shouldn't reach here)
    return 1e30;
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