#version 450

#define NULL_IDX         0xFFFFFFFF
#define MAX_FRAGS        32
#define MAX_GLOBAL_ROOTS 64

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    float normalX;
    float normalY;
    float normalZ;
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
    uint rootTable[];
};

layout(set = 0, binding = 5, std430) buffer DebugInfo {
    uint debugMinMaxPos;
    uint debugNumRoots;
    uint debugPad0;
    uint debugPad1;
};

layout(push_constant) uniform PushConstants {
    float planeThreshold;
    float proximityScale;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
    float nearPlane;
    float farPlane;
    float projX;
    float projY;
    float colorThreshold;
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

uint findMinMaxPosition(out uint rootCount) {
    uint minPos = NULL_IDX;
    rootCount = 0u;
    for (uint i = 0u; i < MAX_GLOBAL_ROOTS; i++) {
        if (rootTable[i] == NULL_IDX) break;
        rootCount++;
        uint pos = rootTable[MAX_GLOBAL_ROOTS + i];
        minPos = min(minPos, pos);
    }
    return minPos;
}

bool isRootInFrontLayer(uint root, uint minMaxPos) {
    for (uint i = 0u; i < MAX_GLOBAL_ROOTS; i++) {
        if (rootTable[i] == NULL_IDX) break;
        if (rootTable[i] == root) {
            return rootTable[MAX_GLOBAL_ROOTS + i] == minMaxPos;
        }
    }
    return false;
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0);
        return;
    }

    uint idx = imageLoad(headPointerImage, coord).r;
    if (idx == NULL_IDX) {
        discard;
    }

    // Compute the global minimum maxPosition
    uint rootCount;
    uint minMaxPos = findMinMaxPosition(rootCount);

    // Write debug info (all pixels write the same value — that's fine)
    // Use atomicMin to avoid partial writes; value is the same from all pixels anyway
    atomicMin(debugMinMaxPos, minMaxPos);
    atomicMax(debugNumRoots, rootCount);

    if (minMaxPos == NULL_IDX) {
        discard;
    }

    uint root = parent[idx];

    if (isRootInFrontLayer(root, minMaxPos)) {
        outColor = unpackColor(nodes[idx].color);
        uint next = nodes[idx].next;
        imageStore(headPointerImage, coord, uvec4(next));
    } else {
        discard;
    }
}