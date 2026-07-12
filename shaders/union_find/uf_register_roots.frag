#version 450

#define NULL_IDX         0xFFFFFFFF
#define MAX_FRAGS        32
#define MAX_GLOBAL_ROOTS 128

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

// Layout: [0..7] = rootIDs, [8..15] = maxDepths
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

void registerRoot(uint root, uint depthBits) {
    for (uint i = 0; i < MAX_GLOBAL_ROOTS; i++) {
        // Try to claim slot i (rootID part)
        uint existing = atomicCompSwap(rootTable[i], NULL_IDX, root);

        if (existing == NULL_IDX || existing == root) {
            // Slot is ours — update max depth (depth stored at offset MAX_GLOBAL_ROOTS)
            atomicMax(rootTable[MAX_GLOBAL_ROOTS + i], depthBits);
            return;
        }
        // Slot taken by different root — try next
    }
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) return;

    uint idx = imageLoad(headPointerImage, coord).r;
    uint count = 0;

    uint seenRoots[MAX_FRAGS];
    uint numSeen = 0;

    while (idx != NULL_IDX && count < MAX_FRAGS) {
        uint root = parent[idx];

        bool alreadySeen = false;
        for (uint i = 0; i < numSeen; i++) {
            if (seenRoots[i] == root) { alreadySeen = true; break; }
        }

        if (!alreadySeen && numSeen < MAX_FRAGS) {
            seenRoots[numSeen] = root;
            numSeen++;
            registerRoot(root, count);
        }

        idx = nodes[idx].next;
        count++;
    }
}