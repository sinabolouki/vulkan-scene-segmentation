#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) flat in uint fragObjectID;

// ═══ PPLL Data Structures ═══

// Head pointer image: each pixel stores index of first node
layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

// Fragment node buffer
struct FragmentNode {
    uint color;
    float depth;
    uint objectID;
    uint next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

// Atomic counter: next free index in node buffer
layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

// ═══ Pack RGBA color into a single uint32 ═══
uint packColor(vec4 color) {
    uint r = uint(clamp(color.r, 0.0, 1.0) * 255.0);
    uint g = uint(clamp(color.g, 0.0, 1.0) * 255.0);
    uint b = uint(clamp(color.b, 0.0, 1.0) * 255.0);
    uint a = uint(clamp(color.a, 0.0, 1.0) * 255.0);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

void main() {
    // ─── Compute shaded color ───
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormal);
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3;
    vec4 finalColor = vec4(fragColor * diffuse, 1.0);

    // ─── Allocate a new node (atomic increment) ───
    uint nodeIndex = atomicAdd(counter, 1u);

    // ─── Overflow check: don't write beyond buffer capacity ───
    uint maxNodes = uint(nodes.length());
    if (nodeIndex >= maxNodes) {
        return; // Buffer full — discard this fragment
    }

    // ─── Insert at head of this pixel's list (atomic exchange) ───
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    uint prevHead = imageAtomicExchange(headPointerImage, pixelCoord, nodeIndex);

    // ─── Write the node data ───
    nodes[nodeIndex].color = packColor(finalColor);
    nodes[nodeIndex].depth = gl_FragCoord.z;  // Depth in [0, 1]
    nodes[nodeIndex].objectID = fragObjectID;
    nodes[nodeIndex].next = prevHead;         // Link to previous head (or 0xFFFFFFFF if first)
}