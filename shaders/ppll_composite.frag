#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D layer0;
layout(set = 0, binding = 1) uniform sampler2D layer1;
layout(set = 0, binding = 2) uniform sampler2D layer2;
layout(set = 0, binding = 3) uniform sampler2D layer3;

layout(push_constant) uniform PushConstants {
    uint numLayers;
    int  displayLayer;  // -1 = composite, 0-3 = show single layer
} pc;

void main() {
    // ─── Single layer debug mode ───
    if (pc.displayLayer >= 0) {
        vec4 c;
        switch (pc.displayLayer) {
            case 0: c = texture(layer0, fragUV); break;
            case 1: c = texture(layer1, fragUV); break;
            case 2: c = texture(layer2, fragUV); break;
            case 3: c = texture(layer3, fragUV); break;
            default: c = vec4(0.0); break;
        }
        // Show on dark background (alpha as visibility)
        vec3 bg = vec3(0.02, 0.02, 0.08);
        outColor = vec4(c.rgb * c.a + bg * (1.0 - c.a), 1.0);
        return;
    }

    // ─── Normal composite: blend back-to-front ───
    vec4 result = vec4(0.02, 0.02, 0.08, 1.0);

    if (pc.numLayers > 3) {
        vec4 c = texture(layer3, fragUV);
        if (c.a > 0.001) result.rgb = c.rgb * c.a + result.rgb * (1.0 - c.a);
    }
    if (pc.numLayers > 2) {
        vec4 c = texture(layer2, fragUV);
        if (c.a > 0.001) result.rgb = c.rgb * c.a + result.rgb * (1.0 - c.a);
    }
    if (pc.numLayers > 1) {
        vec4 c = texture(layer1, fragUV);
        if (c.a > 0.001) result.rgb = c.rgb * c.a + result.rgb * (1.0 - c.a);
    }
    if (pc.numLayers > 0) {
        vec4 c = texture(layer0, fragUV);
        if (c.a > 0.001) result.rgb = c.rgb * c.a + result.rgb * (1.0 - c.a);
    }

    outColor = result;
}