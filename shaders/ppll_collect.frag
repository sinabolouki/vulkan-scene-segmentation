#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) flat in uint fragObjectID;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    uint objectID;
    float nearPlane;
    float farPlane;
} pc;

layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    float depth;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

uint packColor(vec4 color) {
    uint r = uint(clamp(color.r, 0.0, 1.0) * 255.0);
    uint g = uint(clamp(color.g, 0.0, 1.0) * 255.0);
    uint b = uint(clamp(color.b, 0.0, 1.0) * 255.0);
    uint a = uint(clamp(color.a, 0.0, 1.0) * 255.0);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

float linearizeDepth(float z) {
    // Convert from non-linear [0,1] depth buffer to linear world-space distance
    return (pc.nearPlane * pc.farPlane) / (pc.farPlane - z * (pc.farPlane - pc.nearPlane));
}

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3;
    vec4 finalColor = vec4(fragColor * diffuse, 1.0);

    // Allocate node
    uint nodeIndex = atomicAdd(counter, 1u);
    if (nodeIndex >= nodes.length()) {
        return;
    }

    // Insert at head
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    uint prevHead = imageAtomicExchange(headPointerImage, pixelCoord, nodeIndex);

    // Linearize depth: convert to [0, 1] range in LINEAR space
    float linDepth = linearizeDepth(gl_FragCoord.z);
    float normalizedDepth = (linDepth - pc.nearPlane) / (pc.farPlane - pc.nearPlane);

    // Write node
    nodes[nodeIndex].color = packColor(finalColor);
    nodes[nodeIndex].depth = normalizedDepth;  // Linear [0, 1]
    nodes[nodeIndex].objectID = fragObjectID;
    nodes[nodeIndex].next = prevHead;
}