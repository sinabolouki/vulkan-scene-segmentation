#version 450

#define NULL_IDX         0xFFFFFFFF
#define MAX_FRAGS        32
#define MAX_GLOBAL_ROOTS 32

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  normalXY;    // Packed view-space normal (2x float16 via packHalf2x16)
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

layout(set = 0, binding = 4, std430) coherent buffer RootTable {
    uint rootTable[];
};

layout(push_constant) uniform PushConstants {
    float depthThreshold;
    float colorThreshold;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
};

void registerRoot(uint root, uint position) {
    for (uint i = 0; i < MAX_GLOBAL_ROOTS; i++) {
        // Fast non-atomic read first (avoid atomic if possible)
        uint existing = rootTable[i];

        if (existing == root) {
            // Already registered — just update max position
            atomicMax(rootTable[MAX_GLOBAL_ROOTS + i], position);
            return;
        }

        if (existing == NULL_IDX) {
            // Slot is empty — try to claim it atomically
            uint actual = atomicCompSwap(rootTable[i], NULL_IDX, root);
            if (actual == NULL_IDX || actual == root) {
                // Claimed it or someone else registered same root
                atomicMax(rootTable[MAX_GLOBAL_ROOTS + i], position);
                return;
            }
            // Slot was taken by a different root between read and CAS — continue
        }
        // Slot taken by different root — try next
    }
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    // Walk from current head (already advanced by previous peels)
    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];
        registerRoot(root, count);
        count++;
        idx = nodes[idx].next;
    }
}