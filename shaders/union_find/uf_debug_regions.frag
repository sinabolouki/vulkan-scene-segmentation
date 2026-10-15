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
// Mode 6: Absolute maximum fragment rank, with component outlines and legend
// Mode 7: Region rank span (same-root fragment count at this pixel)
// ═══════════════════════════════════════════════════════════════════════════════

#define NULL_IDX 0xFFFFFFFFu

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

// Mode 6 options. These do not require changes to the renderer's push constants.
const float MODE6_LAYER_OPACITY = 0.75;
const bool MODE6_FLAT_COLORS = false;
const bool MODE6_SHOW_LEGEND = true;
const vec3 MODE6_BACKGROUND = vec3(0.02, 0.02, 0.08);

bool mode6Inside(ivec2 coord, ivec2 extent) {
    return all(greaterThanEqual(coord, ivec2(0)))
        && all(lessThan(coord, extent));
}

vec4 mode6Invalid(ivec2 coord) {
    // A checkerboard cannot be confused with an ordinary purple rank tint.
    bool bright = (((coord.x / 6) + (coord.y / 6)) & 1) == 0;
    return vec4(bright ? vec3(1.0, 0.0, 1.0) : vec3(0.08, 0.0, 0.08), 1.0);
}

bool mode6ReadRoot(ivec2 coord, ivec2 extent, uint validNodeCount,
                   out uint node, out uint root, out bool invalid) {
    node = NULL_IDX;
    root = NULL_IDX;
    invalid = false;
    if (!mode6Inside(coord, extent)) return false;

    node = imageLoad(headPointerImage, coord).r;
    if (node == NULL_IDX) return false;
    if (node >= validNodeCount) {
        invalid = true;
        return false;
    }

    root = parent[node];
    if (root >= validNodeCount || root >= uint(rootMaxPos.length())) {
        invalid = true;
        return false;
    }
    // Root registration and this display must agree on compressed root IDs.
    // Report an unfinished parent chain instead of coloring an intermediate node.
    if (parent[root] != root) {
        invalid = true;
        return false;
    }
    return true;
}

// Compact 3-by-5 digits, read from the most significant row first.
const uint MODE6_DIGITS[10] = uint[10](
    0x7b6fu, 0x2c97u, 0x73e7u, 0x73cfu, 0x5bc9u,
    0x79cfu, 0x79efu, 0x7249u, 0x7befu, 0x7bcfu
);

bool mode6DigitPixel(ivec2 coord, uint digit) {
    if (!mode6Inside(coord, ivec2(3, 5))) return false;
    uint bit = uint((4 - coord.y) * 3 + (2 - coord.x));
    return (MODE6_DIGITS[digit] & (1u << bit)) != 0u;
}

bool mode6NumberPixel(ivec2 pixel, uint value) {
    if (any(lessThan(pixel, ivec2(0)))) return false;
    ivec2 coord = pixel / 2;
    if (value < 10u) return mode6DigitPixel(coord, value);
    return mode6DigitPixel(coord, value / 10u)
        || mode6DigitPixel(coord - ivec2(4, 0), value % 10u);
}

bool mode6Legend(ivec2 coord, ivec2 extent, out vec4 color) {
    color = vec4(MODE6_BACKGROUND, 1.0);
    if (!MODE6_SHOW_LEGEND || extent.x < 48 || extent.y < 48) return false;

    // Wrap the 32 numbered swatches on narrower viewports.
    int columns = min(32, (extent.x - 24) / 24);
    int rows = (32 + columns - 1) / columns;
    ivec2 panelSize = ivec2(columns * 24 + 8, rows * 24 + 8);
    ivec2 origin = ivec2(8, extent.y - panelSize.y - 8);
    if (origin.y < 8) return false;

    ivec2 local = coord - origin;
    if (!mode6Inside(local, panelSize)) return false;
    color = vec4(0.015, 0.02, 0.03, 1.0);

    ivec2 grid = local - ivec2(4);
    if (!mode6Inside(grid, ivec2(columns * 24, rows * 24))) return true;
    uint rank = uint((grid.y / 24) * columns + grid.x / 24);
    if (rank >= 32u) return true;

    ivec2 cell = grid % 24;
    if (!mode6Inside(cell, ivec2(22))) return true;

    // The white frame identifies the current global minimum by its number.
    bool nextPeel = rank == globalMinMaxPos;
    int borderWidth = nextPeel ? 2 : 1;
    if (cell.x < borderWidth || cell.y < borderWidth
        || cell.x >= 22 - borderWidth || cell.y >= 22 - borderWidth) {
        // color = vec4(nextPeel ? vec3(1.0) : vec3(0.05), 1.0);
        color = vec4(vec3(0.05), 1.0);
        return true;
    }

    vec3 swatch = LAYER_COLORS[rank];
    int numberWidth = rank < 10u ? 6 : 14;
    ivec2 numberOrigin = ivec2((22 - numberWidth) / 2, 6);
    if (mode6NumberPixel(cell - numberOrigin, rank)) {
        float luminance = dot(swatch, vec3(0.2126, 0.7152, 0.0722));
        swatch = luminance > 0.45 ? vec3(0.02) : vec3(1.0);
    }
    color = vec4(swatch, 1.0);
    return true;
}

vec4 mode6Color(ivec2 coord) {
    ivec2 extent = min(ivec2(screenWidth, screenHeight), imageSize(headPointerImage));
    if (!mode6Inside(coord, extent)) return vec4(MODE6_BACKGROUND, 1.0);

    vec4 legendColor;
    if (mode6Legend(coord, extent, legendColor)) return legendColor;

    // The allocation counter can exceed capacity when collection overflows.
    uint validNodeCount = min(counter,
        min(uint(nodes.length()), uint(parent.length())));

    uint node, root;
    bool invalid;
    if (!mode6ReadRoot(coord, extent, validNodeCount, node, root, invalid)) {
        return invalid ? mode6Invalid(coord) : vec4(MODE6_BACKGROUND, 1.0);
    }

    uint maxRank = rootMaxPos[root];
    uint minRank = globalMinMaxPos;
    if (minRank == NULL_IDX || maxRank < minRank || maxRank >= 32u) {
        return mode6Invalid(coord);
    }

    // Maximum fragment rank is not an actual peel iteration number. Color the
    // absolute rank so an unrelated change in the minimum cannot shift all hues.
    // Rank zero is valid and uses the first palette entry.
    vec3 original = unpackColor(nodes[node].color).rgb;
    float luminance = dot(original, vec3(0.2126, 0.7152, 0.0722));
    vec3 color = MODE6_FLAT_COLORS ? LAYER_COLORS[maxRank]
        : mix(vec3(luminance), LAYER_COLORS[maxRank], MODE6_LAYER_OPACITY);

    bool nextPeel = maxRank == minRank;
    if (nextPeel) color = mix(color, vec3(1.0), 0.12);

    // Different components get a boundary even when their rank colors match.
    // Equality is evaluated within this frame, so allocation ID changes alone
    // cannot create or remove these boundaries.
    const ivec2 offsets[4] = ivec2[4](
        ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1)
    );
    for (int direction = 0; direction < 4; ++direction) {
        uint neighborNode, neighborRoot;
        bool neighborInvalid;
        bool neighborValid = mode6ReadRoot(coord + offsets[direction], extent,
            validNodeCount, neighborNode, neighborRoot, neighborInvalid);
        if (!neighborValid || neighborRoot != root) {
            // White outlines mark next-peel candidates; other outlines are dark.
            // color = nextPeel ? vec3(0.98) : vec3(0.015);
            color = vec3(0.015);
            break;
        }
    }
    return vec4(color, 1.0);
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (coord.x < 0 || coord.y < 0
        || coord.x >= int(screenWidth) || coord.y >= int(screenHeight)) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Mode 6 validates all node and root indices before accessing their buffers.
    if (targetLayer == 6u) {
        outColor = mode6Color(coord);
        return;
    }

    uint idx = imageLoad(headPointerImage, coord).r;

    // No fragment at this pixel
    if (idx == NULL_IDX || idx >= counter) {
        outColor = vec4(0.02, 0.02, 0.08, 1.0);
        return;
    }

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
