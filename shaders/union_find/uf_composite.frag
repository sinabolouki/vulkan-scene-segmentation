#version 450

layout(set = 0, binding = 0) uniform sampler2D layer0;
layout(set = 0, binding = 1) uniform sampler2D layer1;
layout(set = 0, binding = 2) uniform sampler2D layer2;
layout(set = 0, binding = 3) uniform sampler2D layer3;
layout(set = 0, binding = 4) uniform sampler2D layer4;
layout(set = 0, binding = 5) uniform sampler2D layer5;
layout(set = 0, binding = 6) uniform sampler2D layer6;
layout(set = 0, binding = 7) uniform sampler2D layer7;
layout(set = 0, binding = 8) uniform sampler2D layer8;
layout(set = 0, binding = 9) uniform sampler2D layer9;
layout(set = 0, binding = 10) uniform sampler2D layer10;
layout(set = 0, binding = 11) uniform sampler2D layer11;
layout(set = 0, binding = 12) uniform sampler2D layer12;
layout(set = 0, binding = 13) uniform sampler2D layer13;
layout(set = 0, binding = 14) uniform sampler2D layer14;
layout(set = 0, binding = 15) uniform sampler2D layer15;

layout(push_constant) uniform PushConstants {
    uint numLayers;
    int  displayLayer;
};

layout(location = 0) out vec4 outColor;

vec4 sampleLayer(uint index, vec2 uv) {
    if      (index == 0u)  return texture(layer0, uv);
    else if (index == 1u)  return texture(layer1, uv);
    else if (index == 2u)  return texture(layer2, uv);
    else if (index == 3u)  return texture(layer3, uv);
    else if (index == 4u)  return texture(layer4, uv);
    else if (index == 5u)  return texture(layer5, uv);
    else if (index == 6u)  return texture(layer6, uv);
    else if (index == 7u)  return texture(layer7, uv);
    else if (index == 8u)  return texture(layer8, uv);
    else if (index == 9u)  return texture(layer9, uv);
    else if (index == 10u) return texture(layer10, uv);
    else if (index == 11u) return texture(layer11, uv);
    else if (index == 12u) return texture(layer12, uv);
    else if (index == 13u) return texture(layer13, uv);
    else if (index == 14u) return texture(layer14, uv);
    else if (index == 15u) return texture(layer15, uv);

    return vec4(0.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(layer0, 0));

    // Debug: show single layer
    if (displayLayer >= 100) {
        vec4 c = sampleLayer(0u, uv);
        outColor = (c.a < 0.01) ? vec4(0.02, 0.02, 0.08, 1.0) : vec4(c.rgb, 1.0);
        return;
    }

    if (displayLayer >= 0 && displayLayer < int(numLayers)) {
        vec4 c = sampleLayer(uint(displayLayer), uv);
        outColor = (c.a < 0.01) ? vec4(0.02, 0.02, 0.08, 1.0) : vec4(c.rgb, 1.0);
        return;
    }

    // Normal composite: back-to-front
    vec3 result = vec3(0.02, 0.02, 0.08);

    for (uint i = numLayers; i > 0u; i--) {
        vec4 c = sampleLayer(i - 1u, uv);
        if (c.a > 0.01) result = c.rgb;
    }

    outColor = vec4(result, 1.0);
}