#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// All color layer textures
layout(set = 0, binding = 0) uniform sampler2D layer0;
layout(set = 0, binding = 1) uniform sampler2D layer1;
layout(set = 0, binding = 2) uniform sampler2D layer2;
layout(set = 0, binding = 3) uniform sampler2D layer3;

layout(push_constant) uniform PushConstants {
    uint numLayers;
} pc;

void main() {
    // Background color
    vec4 result = vec4(0.02, 0.02, 0.08, 1.0);

    // Blend layers BACK-TO-FRONT (furthest layer first)
    // Under-operator: result = layer_color * layer_alpha + result * (1 - layer_alpha)
    
    // Layer 3 (furthest)
    if (pc.numLayers > 3) {
        vec4 c = texture(layer3, fragUV);
        if (c.a > 0.0) {
            result = vec4(c.rgb * c.a + result.rgb * (1.0 - c.a), 1.0);
        }
    }

    // Layer 2
    if (pc.numLayers > 2) {
        vec4 c = texture(layer2, fragUV);
        if (c.a > 0.0) {
            result = vec4(c.rgb * c.a + result.rgb * (1.0 - c.a), 1.0);
        }
    }

    // Layer 1
    if (pc.numLayers > 1) {
        vec4 c = texture(layer1, fragUV);
        if (c.a > 0.0) {
            result = vec4(c.rgb * c.a + result.rgb * (1.0 - c.a), 1.0);
        }
    }

    // Layer 0 (nearest — on top)
    if (pc.numLayers > 0) {
        vec4 c = texture(layer0, fragUV);
        if (c.a > 0.0) {
            result = vec4(c.rgb * c.a + result.rgb * (1.0 - c.a), 1.0);
        }
    }

    outColor = result;
}