#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Region label readback for the SAM2 vs union-find evaluation.
//
// For depth layer `targetLayer` (the k-th fragment of the sorted list, the same
// fragment sam2_resolve.frag writes into the layer image sent to SAM2), store the
// union-find root of that fragment. 0xFFFFFFFF = no fragment at this depth layer.
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX      0xFFFFFFFFu
#define MAX_FIND_ITER 64u

struct FragmentNode {
    uint  color;
    uint  unshadedColor;
    float depth;
    float normalX;
    float normalY;
    float normalZ;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D headPointerImage;

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) readonly buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) readonly buffer ParentBuffer {
    uint parent[];
};

layout(set = 0, binding = 4, std430) writeonly buffer RegionLabelBuffer {
    uint regionLabels[];
};

layout(push_constant) uniform Push {
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
    uint  mergeMode;
} pc;

uint findRoot(uint x) {
    for (uint i = 0u; i < MAX_FIND_ITER; ++i) {
        uint p = parent[x];
        if (p == x) break;
        x = p;
    }
    return x;
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    if (pix.x >= int(pc.screenWidth) || pix.y >= int(pc.screenHeight)) return;

    uint outIndex = uint(pix.y) * pc.screenWidth + uint(pix.x);
    uint validCount = min(counter, min(uint(nodes.length()), uint(parent.length())));

    uint cur = imageLoad(headPointerImage, pix).r;
    for (uint i = 0u; i < pc.targetLayer && cur != NULL_IDX; ++i) {
        cur = nodes[cur].next;
    }

    regionLabels[outIndex] = (cur == NULL_IDX || cur >= validCount) ? NULL_IDX : findRoot(cur);
}
