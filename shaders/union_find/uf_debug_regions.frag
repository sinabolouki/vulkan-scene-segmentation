#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Debug Region Visualization
//
// Mode 0: Region/component color (hash of root index)
// Mode 1: Object ID color (hash of objectID — ground truth)
// Mode 2: Depth (grayscale)
// Mode 3: View-space normals (RGB)
// Mode 4: Original fragment color
// Mode 5: Per-pixel fragment count heatmap
// Mode 6: Peel layer — rootMaxPos[root] relative to globalMin
// Mode 7: Region rank span (same-root fragment count at this pixel)
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX 0xFFFFFFFFu

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

layout(set = 0, binding = 1, std430) readonly buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) readonly buffer AtomicCounter {
    uint counter;
};

layout(set = 0, binding = 3, std430) readonly buffer ParentBuffer {
    uint parent[];
};

layout(set = 0, binding = 4, std430) readonly buffer RootMaxPosBuffer {
    uint rootMaxPos[];
};

layout(set = 0, binding = 5, std430) readonly buffer GlobalMinBuffer {
    uint globalMinMaxPos;
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

const vec3 LAYER_COLORS[32] = vec3[32](
    vec3(0.90, 0.10, 0.10),  //  0: Red
    vec3(0.10, 0.70, 0.10),  //  1: Green
    vec3(0.20, 0.40, 1.00),  //  2: Blue
    vec3(1.00, 0.85, 0.10),  //  3: Yellow
    vec3(0.80, 0.10, 0.90),  //  4: Purple
    vec3(0.00, 0.85, 0.85),  //  5: Cyan
    vec3(1.00, 0.50, 0.00),  //  6: Orange
    vec3(0.95, 0.40, 0.70),  //  7: Pink
    vec3(0.40, 0.80, 0.20),  //  8: Lime
    vec3(0.55, 0.25, 0.00),  //  9: Brown
    vec3(0.50, 0.50, 1.00),  // 10: Periwinkle
    vec3(1.00, 0.00, 0.50),  // 11: Rose
    vec3(0.00, 0.50, 0.30),  // 12: Teal
    vec3(0.75, 0.75, 0.00),  // 13: Olive
    vec3(0.30, 0.00, 0.60),  // 14: Indigo
    vec3(1.00, 0.70, 0.50),  // 15: Peach
    vec3(0.00, 0.35, 0.70),  // 16: Navy Blue
    vec3(0.70, 1.00, 0.40),  // 17: Yellow-Green
    vec3(0.85, 0.00, 0.40),  // 18: Crimson
    vec3(0.40, 0.90, 0.70),  // 19: Mint
    vec3(0.60, 0.40, 0.20),  // 20: Sienna
    vec3(0.70, 0.50, 1.00),  // 21: Lavender
    vec3(0.00, 0.65, 0.00),  // 22: Forest Green
    vec3(1.00, 0.30, 0.30),  // 23: Coral
    vec3(0.30, 0.70, 1.00),  // 24: Sky Blue
    vec3(0.90, 0.60, 0.00),  // 25: Amber
    vec3(0.50, 0.00, 0.30),  // 26: Maroon
    vec3(0.60, 1.00, 0.80),  // 27: Aquamarine
    vec3(0.80, 0.80, 0.50),  // 28: Khaki
    vec3(0.40, 0.20, 0.80),  // 29: Violet
    vec3(0.00, 0.45, 0.45),  // 30: Dark Cyan
    vec3(1.00, 0.80, 0.80)   // 31: Light Pink
);

// ─── Hash function for root → color ─────────────────────────────────────────

vec3 hashColor(uint id) {
    uint x = id;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;

    uint r = (x & 0xFFu);
    uint g = ((x >> 8u) & 0xFFu);
    uint b = ((x >> 16u) & 0xFFu);

    vec3 color = vec3(float(r), float(g), float(b)) / 255.0;
    float maxC = max(color.r, max(color.g, color.b));
    if (maxC < 0.3) color += 0.3;
    color = normalize(color) * 0.8 + 0.1;

    return clamp(color, 0.0, 1.0);
}

vec4 unpackColor(uint packed) {
    return vec4(
        float(packed & 0xFFu) / 255.0,
        float((packed >> 8u) & 0xFFu) / 255.0,
        float((packed >> 16u) & 0xFFu) / 255.0,
        float((packed >> 24u) & 0xFFu) / 255.0
    );
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint idx = imageLoad(headPointerImage, coord).r;

    // No fragment at this pixel
    if (idx == NULL_IDX || idx >= counter) {
        outColor = vec4(0.02, 0.02, 0.08, 1.0);
        return;
    }

    uint validCount = min(counter, uint(rootMaxPos.length()));
    uint root = parent[idx];

    // ─── Visualization modes ─────────────────────────────────────────────

    if (targetLayer == 0u) {
        // Mode 0: Region/component color (hash of root index)
        outColor = vec4(hashColor(root), 1.0);
    }
    else if (targetLayer == 1u) {
        // Mode 1: Object ID color (hash of objectID)
        outColor = vec4(hashColor(nodes[idx].objectID), 1.0);
    }
    else if (targetLayer == 2u) {
        // Mode 2: Depth visualization (grayscale)
        float d = nodes[idx].depth;
        outColor = vec4(vec3(1.0 - d), 1.0);
    }
    else if (targetLayer == 3u) {
        // Mode 3: View-space normal visualization
        vec3 n = normalize(vec3(nodes[idx].normalX, nodes[idx].normalY, nodes[idx].normalZ));
        outColor = vec4(n * 0.5 + 0.5, 1.0);
    }
    else if (targetLayer == 4u) {
        // Mode 4: Original fragment color
        outColor = unpackColor(nodes[idx].color);
    }
    else if (targetLayer == 5u) {
        // Mode 5: Fragment count heatmap (total fragments at this pixel)
        uint count = 0u;
        uint walk = idx;
        while (walk != NULL_IDX && count < 32u) {
            count++;
            walk = nodes[walk].next;
        }
        float t = clamp(float(count) / 16.0, 0.0, 1.0);
        vec3 heat;
        if (t < 0.5) {
            heat = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t * 2.0);
        } else {
            heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
        }
        outColor = vec4(heat, 1.0);
    }
    else if (targetLayer == 6u) {
        // Mode 6: Peel layer visualization
        // Shows which peel iteration would output this pixel's front root.
        //
        // rootMaxPos[root] = the worst-case (deepest) position of this root
        // globalMinMaxPos  = the smallest maxPos among all visible roots
        // layerIdx         = how many peel iterations before this root gets output
        //
        // Layer 0 (red)  = peels first
        // Layer 1 (green) = peels second
        // White = root out of bounds

        if (root >= validCount) {
            outColor = vec4(1.0, 1.0, 1.0, 1.0);  // White = invalid root
            return;
        }

        uint maxPos = rootMaxPos[root];
        uint minPos = globalMinMaxPos;

        if (maxPos == 0u && minPos == 0u) {
            // Unwritten root (rootMaxPos was cleared to 0 and never atomicMax'd)
            // This means this root has no registered fragments — shouldn't happen
            // but handle gracefully
            outColor = vec4(0.3, 0.3, 0.3, 1.0);  // Gray = unregistered
        }
        else if (minPos == NULL_IDX) {
            // No roots registered at all
            outColor = vec4(0.5, 0.0, 0.5, 1.0);  // Dark magenta = empty scene
        }
        else {
            uint layerIdx = maxPos - minPos;
            layerIdx = min(layerIdx, 31u);
            outColor = vec4(LAYER_COLORS[layerIdx], 1.0);
        }
    }
    else if (targetLayer == 7u) {
        // Mode 7: Region rank span (how many fragments at this pixel share front root)
        uint frontRoot = root;
        uint sameRootCount = 0u;
        uint walk = idx;
        while (walk != NULL_IDX && walk < counter && sameRootCount < 32u) {
            if (parent[walk] == frontRoot) {
                sameRootCount++;
            }
            walk = nodes[walk].next;
        }

        if (sameRootCount <= 1u) {
            outColor = vec4(0.1, 0.2, 0.9, 1.0);   // Blue — ideal
        } else if (sameRootCount == 2u) {
            outColor = vec4(0.0, 0.8, 0.8, 1.0);   // Cyan
        } else if (sameRootCount == 3u) {
            outColor = vec4(0.2, 0.9, 0.2, 1.0);   // Green
        } else if (sameRootCount == 4u) {
            outColor = vec4(0.9, 0.9, 0.1, 1.0);   // Yellow
        } else {
            outColor = vec4(1.0, 0.1, 0.1, 1.0);   // Red — bowl problem
        }
    }
    else {
        outColor = vec4(hashColor(root), 1.0);
    }
}