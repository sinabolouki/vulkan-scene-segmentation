#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Union-Find Merge Pass — Soft-Cost Connectivity (4-neighborhood)
//
// Instead of two disjoint hard tests (tangent-plane OR proximity), all signals
// are combined into a single scalar cost:
//
//   cost = w_plane  * planeResidual  / (planeThreshold  * pixelStep)
//        + w_normal * (1 - normalDot) * 0.5
//        + w_depth  * |depthA-depthB| / (proximityScale * pixelStep)
//
// Each term is normalized so that "1.0" means "right at the threshold".
// Two fragments are merged if cost < 1.0.
//
// The neighborhood is the symmetric 4-set { N, W, NW, NE } — every adjacent
// pair is still considered exactly once due to union symmetry.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX       0xFFFFFFFFu
#define MAX_FRAGS      32u
#define MAX_FIND_ITER  64u

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
    float planeThreshold;    // Reference plane-residual scale, e.g. 0.05
    float proximityScale;    // Reference proximity scale, e.g. 3.0
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;       // Unused in merge
    float nearPlane;
    float farPlane;
    float projX;             // projection[0][0]
    float projY;             // projection[1][1]
    float colorThreshold;
} pc;

// ─── Soft-cost weights (tune here or promote to push constants) ─────────────
const float W_PLANE       = 1.0;
const float W_NORMAL      = 1.0;
const float W_DEPTH       = 1.0;
const float W_COLOR       = 0.5; 
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

// ─── Color Helpers (unused right now, kept for future re-enable) ────────────

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

// ─── Soft-Cost Connectivity ─────────────────────────────────────────────────
//
// Returns a normalized cost. cost < COST_THRESHOLD (= 1.0) means "accept".
// A rejected pair returns 1e30 so it can never win best-neighbor selection.

float connectionCost(uint idxA, uint idxB, ivec2 pixelA, ivec2 pixelB) {
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

    // ── Term 4: color dissimilarity ──────────────────────────────────────
    float colDist = colorDistance(nodes[idxA].color, nodes[idxB].color);
    float colorTerm = colDist / max(pc.colorThreshold, 1e-6);

    // ── Combined cost ────────────────────────────────────────────────────
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

    // ─── Load local fragments ─────────────────────────────────────────────
    uint localFragments[MAX_FRAGS];
    uint localCount = 0u;

    uint index = imageLoad(headPointerImage, coordinate).r;
    while (validNodeIndex(index, validNodeCount) && localCount < MAX_FRAGS) {
        localFragments[localCount++] = index;
        index = nodes[index].next;
    }

    if (localCount == 0u) return;

    // ─── Symmetric 4-neighborhood: N, W, NW, NE ──────────────────────────
    // Every adjacent pair is visited exactly once because unions are
    // symmetric — the "other" pixel of the pair would only add duplicates.
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

        // For each local fragment, find the best (lowest-cost) connectable neighbor
        for (uint local = 0u; local < localCount; ++local) {

            uint  bestNeighbor = NULL_IDX;
            float bestCost     = COST_THRESHOLD;   // any accepted cost is < this

            uint neighborIndex = imageLoad(headPointerImage, neighborCoordinate).r;
            uint neighborCount = 0u;

            while (validNodeIndex(neighborIndex, validNodeCount) &&
                   neighborCount < MAX_FRAGS) {

                float cost = connectionCost(localFragments[local], neighborIndex,
                                            coordinate, neighborCoordinate);

                if (cost < bestCost) {
                    bestCost     = cost;
                    bestNeighbor = neighborIndex;
                }

                neighborIndex = nodes[neighborIndex].next;
                ++neighborCount;
            }

            if (bestNeighbor != NULL_IDX) {
                unionSets(localFragments[local], bestNeighbor);
            }
        }
    }
}