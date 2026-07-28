#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 view;        // ← NEW
    uint objectID;
    float nearPlane;
    float farPlane;
} pc;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormalView;  // ← renamed for clarity: VIEW-space normal
layout(location = 2) flat out uint fragObjectID;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;

    // fragColor = vec3(0.5f,0.5f,0.5f); 
    fragColor = inColor;
    // Transform normal to VIEW space (not world space)
    // For uniform scale: mat3(view * model) is sufficient
    // For non-uniform scale you'd need inverse-transpose, but for a thesis prototype this is fine
    mat3 normalMatrix = mat3(pc.view * pc.model);
    fragNormalView = normalize(normalMatrix * inNormal);

    fragObjectID = 0;
}