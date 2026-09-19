#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Resolve Pass — Uses per-root maxPosition buffer + global min
//
// For each pixel:
//   1. Read globalMinMaxPos (computed by find_min pass)
//   2. Check if front fragment's root has rootMaxPos[root] == globalMin
//   3. If yes: output color, advance head
//   4. If no: discard (leave for future peel)
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX 0xFFFFFFFF

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    uint  unshadedColor;  // Packed RGBA8 (unlit base color)
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

layout(set = 0, binding = 4, std430) readonly buffer RootMaxPosBuffer {
    uint rootMaxPos[];
};

layout(set = 0, binding = 5, std430) readonly buffer GlobalMinBuffer {
    uint globalMinMaxPos;
};

layout(set = 0, binding = 6, std430) buffer DebugInfo {
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

vec4 unpackColor(uint packed) {
    return vec4(
        float(packed & 0xFFu) / 255.0,
        float((packed >> 8u) & 0xFFu) / 255.0,
        float((packed >> 16u) & 0xFFu) / 255.0,
        float((packed >> 24u) & 0xFFu) / 255.0
    );
}

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

    uint minMaxPos = globalMinMaxPos;
    if (minMaxPos == NULL_IDX) {
        // No roots registered at all
        discard;
    }

    // Write debug info (same value from all pixels — one atomic is fine)
    atomicMin(debugMinMaxPos, minMaxPos);

    uint root = parent[idx];
    uint validCount = min(counter, uint(rootMaxPos.length()));

    if (root < validCount && rootMaxPos[root] == minMaxPos) {
        // This root is in the front layer — output it
        outColor = unpackColor(nodes[idx].color);

        // Advance head past this fragment
        imageStore(headPointerImage, coord, uvec4(nodes[idx].next));
    } else {
        // Not in front layer — leave for later peel
        discard;
    }
}