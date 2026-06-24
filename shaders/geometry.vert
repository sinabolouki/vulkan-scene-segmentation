#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    uint objectID;
} pc;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;
layout(location = 2) flat out uint fragObjectID;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;

    fragNormal = mat3(pc.model) * inNormal;
    fragUV = inUV;
    fragObjectID = pc.objectID;
}