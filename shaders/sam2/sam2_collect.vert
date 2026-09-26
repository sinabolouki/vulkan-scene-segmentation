#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    uint objectID;
    float nearPlane;
    float farPlane;
    int  textureIndex;
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vNormalView;
layout(location = 3) out float vDepthLinear;

void main()
{
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    vec4 viewPos  = pc.view  * worldPos;

    vNormalView   = normalize(mat3(pc.view * pc.model) * inNormal);
    vColor        = inColor;
    vUV           = inUV;

    // Linear depth in [0,1]: 0 at near plane, 1 at far plane.
    float linear01 = (-viewPos.z - pc.nearPlane) / (pc.farPlane - pc.nearPlane);
    vDepthLinear   = clamp(linear01, 0.0, 1.0);

    gl_Position = pc.viewProj * worldPos;
}