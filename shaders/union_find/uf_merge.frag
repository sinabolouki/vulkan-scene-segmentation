#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Union-Find Merge Pass — Dual-Criterion Connectivity
//
// Two fragments are merged if EITHER:
//   1. Their normals are compatible AND each lies on the other's tangent plane
//   2. Their normals are incompatible (crease) AND they are spatially close
//
// This avoids the pixel-footprint estimation entirely for the primary test,
// eliminating screen-edge and near-camera artifacts.
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
    float planeThreshold;    // Max tangent-plane residual (world units), e.g. 0.05
    float proximityScale;    // Multiplier for proximity fallback, e.g. 3.0
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;       // Unused in merge
    float nearPlane;
    float farPlane;
    float projX;             // projection[0][0]
    float projY;             // projection[1][1]
} pc;

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

// Compute the actual 3D distance between two pixel centers at a reference depth.
// This replaces the broken viewPixelSize() and is correct at all screen positions.
float computePixelStep(ivec2 pixelA, ivec2 pixelB, float refDepth) {
    vec3 refA = reconstructViewPosition(pixelA, refDepth);
    vec3 refB = reconstructViewPosition(pixelB, refDepth);
    return length(refB - refA);
}

// ─── Connectivity Decision ───────────────────────────────────────────────────

// Returns true if the two fragments should be connected.
bool shouldConnect(uint idxA, uint idxB, ivec2 pixelA, ivec2 pixelB) {
    float depthA = nodes[idxA].depth;
    float depthB = nodes[idxB].depth;

    vec3 posA = reconstructViewPosition(pixelA, depthA);
    vec3 posB = reconstructViewPosition(pixelB, depthB);
    vec3 delta = posB - posA;

    vec3 normalA = getNodeNormal(idxA);
    vec3 normalB = getNodeNormal(idxB);

    float normalDot = dot(normalA, normalB);

    // ─── Adaptive threshold from actual pixel spacing ─────────────────────
    // Use average depth as reference to get the local pixel footprint.
    // This is correct at screen edges because it uses reconstructViewPosition.
    float refDepth = 0.5 * (depthA + depthB);
    float pixelStep = computePixelStep(pixelA, pixelB, refDepth);

    // ─── Case 1: Compatible normals → tangent-plane test ──────────────────
    if (normalDot > 0.25) {
        // How far is B from A's tangent plane?
        float residualA = abs(dot(normalA, delta));
        // How far is A from B's tangent plane?
        float residualB = abs(dot(normalB, -delta));

        // Use the larger residual (more conservative)
        float maxResidual = max(residualA, residualB);

        // Threshold: planeThreshold scaled by the pixel step.
        // A flat surface has residual ≈ 0 regardless of angle.
        // A curved surface has residual ∝ curvature × pixelStep².
        // We allow a small multiple of pixelStep to handle mild curvature.
        float threshold = pc.planeThreshold * pixelStep;

        return maxResidual < threshold;
    }

    // ─── Case 2: Incompatible normals → proximity test ────────────────────
    // This handles sharp creases (cube edges, hard surface joins).
    // Fragments at a shared edge are at essentially the same position.
    else {
        float dist = length(delta);

        // Allow a more generous distance for creases.
        // proximityScale controls how many "pixel steps" apart they can be.
        float threshold = pc.proximityScale * pixelStep;

        return dist < threshold;
    }
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

    // ─── Compare with 8 neighbors ────────────────────────────────────────
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            if (offsetX == 0 && offsetY == 0) continue;

            ivec2 neighborCoordinate = coordinate + ivec2(offsetX, offsetY);

            if (neighborCoordinate.x < 0 || neighborCoordinate.y < 0 ||
                neighborCoordinate.x >= int(pc.screenWidth) ||
                neighborCoordinate.y >= int(pc.screenHeight)) {
                continue;
            }

            // For each local fragment, find the best connectable neighbor
            for (uint local = 0u; local < localCount; ++local) {

                uint bestNeighbor = NULL_IDX;
                float bestDist = 1e30;

                uint neighborIndex = imageLoad(headPointerImage, neighborCoordinate).r;
                uint neighborCount = 0u;

                while (validNodeIndex(neighborIndex, validNodeCount) &&
                       neighborCount < MAX_FRAGS) {

                    if (shouldConnect(localFragments[local], neighborIndex,
                                      coordinate, neighborCoordinate)) {
                        // Among all connectable neighbors, pick the closest in depth
                        // to avoid merging across layers when multiple candidates pass
                        float dA = nodes[localFragments[local]].depth;
                        float dB = nodes[neighborIndex].depth;
                        float depthDist = abs(dA - dB);

                        if (depthDist < bestDist) {
                            bestDist = depthDist;
                            bestNeighbor = neighborIndex;
                        }
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
}