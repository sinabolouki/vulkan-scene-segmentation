#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormalView;
layout(location = 2) flat out uint fragObjectID;
layout(location = 3) out vec2 fragUV;
layout(location = 4) out vec3 fragPositionView;

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
    fragPositionView = (pc.view * worldPos).xyz;

    // Shading normal in view space. Geometry normals are derived per fragment.
    mat3 normalMatrix = transpose(inverse(mat3(pc.view * pc.model)));
    vec3 normalView = normalMatrix * inNormal;
    float normalLength2 = dot(normalView, normalView);
    fragNormalView = normalLength2 > 0.0
        ? normalView * inversesqrt(normalLength2)
        : vec3(0.0);

    fragColor = inColor;
    fragObjectID = pc.objectID;
    fragUV = inUV;
}
