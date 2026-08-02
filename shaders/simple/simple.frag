#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    int  textureIndex;
} pc;

// Texture array (set 0, binding 0)
layout(set = 0, binding = 0) uniform sampler2D textures[64];

void main() {
    // Base color from vertex (material Kd)
    vec3 baseColor = fragColor;

    // Sample texture if available
    if (pc.textureIndex >= 0) {
        vec4 texColor = texture(textures[pc.textureIndex], fragUV);

        // Alpha test (Sponza leaves, chains, etc.)
        if (texColor.a < 0.5) {
            discard;
        }

        baseColor *= texColor.rgb;
    }

    // Simple directional light
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 normal = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3;

    outColor = vec4(baseColor * diffuse, 1.0);
}