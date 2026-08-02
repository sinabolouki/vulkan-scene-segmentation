#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormalView;
layout(location = 2) flat out uint fragObjectID;
layout(location = 3) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    uint objectID;
    float nearPlane;
    float farPlane;
    int  textureIndex;
} pc;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;

    // Normal in view space
    mat3 normalMatrix = transpose(inverse(mat3(pc.view * pc.model)));
    fragNormalView = normalize(normalMatrix * inNormal);

    fragColor = inColor;
    fragObjectID = pc.objectID;
    fragUV = inUV;
}