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
    uint  normalXY;    // packHalf2x16(viewNormal.xy)
    uint  objectID;    // Kept only because it is part of the collect-pass layout
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

// The collect shader stores only normal.x and normal.y. Therefore the sign of
// normal.z cannot be recovered. This pass only needs |normal.z| to estimate the
// grazing angle, so reconstructing the positive magnitude is sufficient.
vec3 unpackNormalWithAbsZ(uint packed) {
    vec2 nxy = unpackHalf2x16(packed);

    float xyLengthSquared = dot(nxy, nxy);
    if (xyLengthSquared > 1.0) {
        nxy *= inversesqrt(xyLengthSquared);
        xyLengthSquared = 1.0;
    }

    float absNz = sqrt(max(1.0 - xyLengthSquared, 0.0));
    return vec3(nxy, absNz);
}

float viewDistanceFromStoredDepth(float normalizedLinearDepth) {
    return pc.nearPlane
         + clamp(normalizedLinearDepth, 0.0, 1.0)
         * (pc.farPlane - pc.nearPlane);
}

// Reconstructs a view-space position from pixel centre and stored linear depth.
// A negative-height Vulkan viewport may mirror the y coordinate. That mirror
// does not affect Euclidean distances as long as both samples use this same
// reconstruction convention.
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
    vec2 pixelSize = viewPixelSize(averageDistance);
    float projectedStep = length(vec2(pixelOffset) * pixelSize);
    projectedStep = max(projectedStep, 1e-7);

    vec3 normalA = unpackNormalWithAbsZ(nodes[idxA].normalXY);
    vec3 normalB = unpackNormalWithAbsZ(nodes[idxB].normalXY);

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
        8.0 * projectedStep
    );

    // Preserve the old depthThreshold control as a small amount of additional
    // view-space slack, but cap it so a large far/near range cannot merge
    // distant objects accidentally.
    float configuredSlack = max(pc.depthThreshold, 0.0)
                          * (pc.farPlane - pc.nearPlane);
    allowedGap += min(configuredSlack, 2.0 * projectedStep);

    return positionGap / max(allowedGap, 1e-7);
}

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
    // candidate in each neighboring list. The previous shader unioned every
    // pair passing a loose test, which could bridge different depth layers.
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
