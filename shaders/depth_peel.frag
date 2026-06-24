#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 fragUV;
layout(location = 3) flat in uint fragObjectID;

layout(set = 0, binding = 0) uniform sampler2D prevDepthTexture;

layout(location = 0) out vec4 outColor;

void main() {
    // Get screen-space UV from fragment coordinates
    ivec2 texSize = textureSize(prevDepthTexture, 0);
    vec2 texCoord = gl_FragCoord.xy / vec2(texSize);

    // Read depth from previous layer
    float prevDepth = texture(prevDepthTexture, texCoord).r;

    // PEEL: discard if this fragment is at same depth or closer than previous layer
    if (gl_FragCoord.z <= prevDepth) {
        discard;
    }

    // Use vertex color from OBJ file
    vec3 baseColor = fragColor;

    // Simple diffuse lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3;

    outColor = vec4(baseColor * diffuse, 1.0);
}