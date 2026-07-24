#version 450

// Union-find merge pass for per-pixel fragment lists.
//
// Connectivity is based primarily on reconstructed VIEW-SPACE POSITION.
// Normal discontinuities are not treated as object boundaries: two cube faces
// may therefore merge across a sharp crease when their samples meet in 3D.
// Normals are used only to estimate how far neighboring samples may be apart
// on a grazing-angle surface.

#define NULL_IDX       0xFFFFFFFFu
#define MAX_FRAGS      32u
#define MAX_FIND_ITER  64u
#define HUGE_COST      3.402823466e+38

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
    float depthThreshold; // Existing normalized-linear-depth tolerance
    float colorThreshold; // Unused; retained to preserve the host-side layout
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;    // Unused; retained to preserve the host-side layout
    float nearPlane;
    float farPlane;
    float projX;          // projection[0][0]
    float projY;          // projection[1][1]
} pc;

// ─── Normal Reading ──────────────────────────────────────────────────────────

vec3 getNodeNormal(uint idx) {
    vec3 n = vec3(nodes[idx].normalX, nodes[idx].normalY, nodes[idx].normalZ);
    float len2 = dot(n, n);
    if (len2 < 1e-12) {
        return vec3(0.0, 0.0, -1.0); // fallback: facing camera
    }
    return n * inversesqrt(len2);
}

// ─── View-Space Reconstruction ───────────────────────────────────────────────

float viewDistanceFromStoredDepth(float normalizedLinearDepth) {
    return pc.nearPlane
         + clamp(normalizedLinearDepth, 0.0, 1.0)
         * (pc.farPlane - pc.nearPlane);
}

vec3 reconstructViewPosition(ivec2 pixel, float normalizedLinearDepth) {
    float viewDistance = viewDistanceFromStoredDepth(normalizedLinearDepth);

    vec2 uv = (vec2(pixel) + vec2(0.5))
            / vec2(float(pc.screenWidth), float(pc.screenHeight));

    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;

    float safeProjX = max(abs(pc.projX), 1e-6);
    float safeProjY = max(abs(pc.projY), 1e-6);

    return vec3(
        ndcX * viewDistance / safeProjX,
        ndcY * viewDistance / safeProjY,
        -viewDistance
    );
}

vec2 viewPixelSize(float viewDistance) {
    return vec2(
        2.0 * viewDistance
            / (float(pc.screenWidth) * max(abs(pc.projX), 1e-6)),
        2.0 * viewDistance
            / (float(pc.screenHeight) * max(abs(pc.projY), 1e-6))
    );
}

// ─── Connection Cost ─────────────────────────────────────────────────────────
//
// Returns a normalized cost. Values <= 1 are considered connected.
//
// The allowed gap starts from the view-space footprint of the actual pixel
// offset and is enlarged at grazing angles. It is capped to avoid connecting
// unrelated surfaces merely because one normal is almost perpendicular to the
// view direction.

float connectionCost(
    uint idxA,
    uint idxB,
    ivec2 pixelA,
    ivec2 pixelB)
{
    float depthA = nodes[idxA].depth;
    float depthB = nodes[idxB].depth;

    vec3 positionA = reconstructViewPosition(pixelA, depthA);
    vec3 positionB = reconstructViewPosition(pixelB, depthB);
    vec3 delta = positionB - positionA;

    float positionGap = length(delta);

    float distanceA = -positionA.z;
    float distanceB = -positionB.z;
    float averageDistance = 0.5 * (distanceA + distanceB);

    ivec2 pixelOffset = pixelB - pixelA;
    // vec2 pixelSize = viewPixelSize(averageDistance);
    // float projectedStep = length(vec2(pixelOffset) * pixelSize);
    // projectedStep = max(projectedStep, 1e-7);
    vec3 posA_avg = reconstructViewPosition(pixelA, 0.5 * (depthA + depthB));
    vec3 posB_avg = reconstructViewPosition(pixelB, 0.5 * (depthA + depthB));
    float projectedStep = length(posB_avg - posA_avg);

    vec3 normalA = getNodeNormal(idxA);
    vec3 normalB = getNodeNormal(idxB);

    // Use abs(nz) for grazing angle estimation.
    // Now we have the true signed nz, but the grazing formula only needs magnitude.
    // 0.25 limits the grazing-angle amplification to 4x per surface.
    float expectedStepA = projectedStep / max(abs(normalA.z), 0.25);
    float expectedStepB = projectedStep / max(abs(normalB.z), 0.25);

    // Use the more permissive of the two surfaces so that a sharp cube crease
    // is not rejected simply because the normals differ by 90 degrees.
    float allowedGap = 2.25 * max(expectedStepA, expectedStepB);

    // Keep the tolerance within a sensible screen-space range.
    allowedGap = clamp(
        allowedGap,
        1.5 * projectedStep,
        12.0 * projectedStep
    );

    return positionGap / max(allowedGap, 1e-7);
}

// ─── Union-Find ──────────────────────────────────────────────────────────────

uint findRoot(uint x) {
    for (uint i = 0u; i < MAX_FIND_ITER; ++i) {
        uint p = parent[x];
        if (p == x) {
            break;
        }
        x = p;
    }
    return x;
}

void unionSets(uint a, uint b) {
    for (uint iteration = 0u; iteration < MAX_FIND_ITER; ++iteration) {
        a = findRoot(a);
        b = findRoot(b);

        if (a == b) {
            return;
        }

        // Parent indices only decrease. This prevents cycles even when several
        // fragment invocations perform unions concurrently.
        uint high = max(a, b);
        uint low  = min(a, b);

        uint previous = atomicMin(parent[high], low);
        if (previous == high || previous == low) {
            return;
        }
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
    if (validNodeCount == 0u) {
        return;
    }

    uint localFragments[MAX_FRAGS];
    uint localCount = 0u;

    uint index = imageLoad(headPointerImage, coordinate).r;
    while (validNodeIndex(index, validNodeCount) && localCount < MAX_FRAGS) {
        localFragments[localCount++] = index;
        index = nodes[index].next;
    }

    if (localCount == 0u) {
        return;
    }

    // For each local fragment, choose only its best geometrically continuous
    // candidate in each neighboring list.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            if (offsetX == 0 && offsetY == 0) {
                continue;
            }

            ivec2 neighborCoordinate =
                coordinate + ivec2(offsetX, offsetY);

            if (neighborCoordinate.x < 0 || neighborCoordinate.y < 0 ||
                neighborCoordinate.x >= int(pc.screenWidth) ||
                neighborCoordinate.y >= int(pc.screenHeight)) {
                continue;
            }

            for (uint local = 0u; local < localCount; ++local) {
                uint bestNeighbor = NULL_IDX;
                float bestCost = HUGE_COST;

                uint neighborIndex =
                    imageLoad(headPointerImage, neighborCoordinate).r;
                uint neighborCount = 0u;

                while (validNodeIndex(neighborIndex, validNodeCount) &&
                       neighborCount < MAX_FRAGS) {
                    float cost = connectionCost(
                        localFragments[local],
                        neighborIndex,
                        coordinate,
                        neighborCoordinate
                    );

                    if (cost < bestCost) {
                        bestCost = cost;
                        bestNeighbor = neighborIndex;
                    }

                    neighborIndex = nodes[neighborIndex].next;
                    ++neighborCount;
                }

                if (bestNeighbor != NULL_IDX && bestCost <= 1.0) {
                    unionSets(localFragments[local], bestNeighbor);
                }
            }
        }
    }
}