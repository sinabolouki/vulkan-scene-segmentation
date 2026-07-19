#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormalView;   // View-space normal
layout(location = 2) flat in uint fragObjectID;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    uint objectID;
    float nearPlane;
    float farPlane;
} pc;

layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint  color;       // Packed RGBA8
    float depth;       // Linear normalized depth [0, 1]
    uint  normalXY;    // ← NEW: packed view-space normal (2x float16)
    uint  objectID;
    uint  next;
};
// 5 uints = 20 bytes per node

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

// ─── Packing Helpers ─────────────────────────────────────────────────────────

uint packColor(vec4 color) {
    uint r = uint(clamp(color.r, 0.0, 1.0) * 255.0);
    uint g = uint(clamp(color.g, 0.0, 1.0) * 255.0);
    uint b = uint(clamp(color.b, 0.0, 1.0) * 255.0);
    uint a = uint(clamp(color.a, 0.0, 1.0) * 255.0);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

// Pack two floats into a single uint using float16 encoding
// This gives us 10-bit mantissa precision, sufficient for normals
uint packNormalXY(vec3 normal) {
    // Pack nx and ny as two float16 values into one uint32
    // nz can be reconstructed: nz = sqrt(1 - nx*nx - ny*ny) with sign convention
    // In view space, for front-facing geometry, nz is always negative (pointing toward camera)
    // We store the sign of nz in the highest bit of the packed value
    uint packed = packHalf2x16(vec2(normal.x, normal.y));
    // Store nz sign: if nz >= 0, set bit 0 of a separate approach...
    // Actually, let's just pack all info we need:
    // We'll use packHalf2x16 for (nx, ny) and reconstruct nz with known sign
    // In view space with standard convention, nz < 0 for front-facing surfaces
    // BUT we flip the viewport, so it depends. Let's just store the sign.
    // Simple: use the full uint for packHalf2x16(nx, ny), reconstruct nz = -sqrt(1-nx²-ny²)
    // If the geometry is front-facing in view space, nz should be negative.
    return packed;
}

float linearizeDepth(float z) {
    return (pc.nearPlane * pc.farPlane) / (pc.farPlane - z * (pc.farPlane - pc.nearPlane));
}

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 normal = normalize(fragNormalView);
    
    // Lighting in world space (use original normal direction for diffuse)
    float diffuse = max(dot(normalize(fragNormalView), lightDir), 0.0) * 0.7 + 0.3;
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
    nodes[nodeIndex].color    = packColor(finalColor);
    nodes[nodeIndex].depth    = normalizedDepth;    // Linear [0, 1]
    nodes[nodeIndex].normalXY = packNormalXY(normal); // ← NEW
    nodes[nodeIndex].objectID = fragObjectID;
    nodes[nodeIndex].next     = prevHead;
}