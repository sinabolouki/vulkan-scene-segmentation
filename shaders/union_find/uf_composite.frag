#version 450

// ═══════════════════════════════════════════════════════════════════════════════
// Composite Pass
//
// Blends layer textures back-to-front onto the swapchain.
// Supports debug display modes.
//
// displayLayer:
//   -1    → normal composite (all layers back-to-front)
//   0-3   → show single layer only
//   >=100 → debug mode (show layer 0 which contains debug visualization)
// ═══════════════════════════════════════════════════════════════════════════════

layout(set = 0, binding = 0) uniform sampler2D layer0;
layout(set = 0, binding = 1) uniform sampler2D layer1;
layout(set = 0, binding = 2) uniform sampler2D layer2;
layout(set = 0, binding = 3) uniform sampler2D layer3;

layout(push_constant) uniform PushConstants {
    uint numLayers;
    int  displayLayer;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(layer0, 0));

    // Debug mode: just show layer 0 directly
    if (displayLayer >= 100) {
        vec4 c = texture(layer0, uv);
        if (c.a < 0.01) {
            outColor = vec4(0.02, 0.02, 0.08, 1.0);  // background
        } else {
            outColor = vec4(c.rgb, 1.0);
        }
        return;
    }

    // Single layer display mode
    if (displayLayer >= 0 && displayLayer < int(numLayers)) {
        vec4 c;
        switch (displayLayer) {
            case 0: c = texture(layer0, uv); break;
            case 1: c = texture(layer1, uv); break;
            case 2: c = texture(layer2, uv); break;
            case 3: c = texture(layer3, uv); break;
            default: c = vec4(0.0); break;
        }
        if (c.a < 0.01) {
            outColor = vec4(0.02, 0.02, 0.08, 1.0);  // background
        } else {
            outColor = vec4(c.rgb, 1.0);
        }
        return;
    }

    // Normal composite: blend layers back-to-front
    // Layer 3 (deepest) → Layer 0 (frontmost)
    vec4 result = vec4(0.02, 0.02, 0.08, 1.0);  // background color

    // Back to front: 3, 2, 1, 0
    vec4 c3 = texture(layer3, uv);
    vec4 c2 = texture(layer2, uv);
    vec4 c1 = texture(layer1, uv);
    vec4 c0 = texture(layer0, uv);

    // Alpha blending (over operator): front over back
    if (c3.a > 0.01) result = vec4(c3.rgb, 1.0);
    if (c2.a > 0.01) result = vec4(c2.rgb, 1.0);
    if (c1.a > 0.01) result = vec4(c1.rgb, 1.0);
    if (c0.a > 0.01) result = vec4(c0.rgb, 1.0);

    outColor = result;
}