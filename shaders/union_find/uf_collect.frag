#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormalView;
layout(location = 2) flat in uint fragObjectID;
layout(location = 3) in vec2 fragUV;
layout(location = 4) in vec3 fragPositionView;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    uint objectID;
    float nearPlane;
    float farPlane;
    int  textureIndex;
} pc;

layout(set = 0, binding = 0, r32ui) uniform coherent uimage2D headPointerImage;

struct FragmentNode {
    uint  color;
    uint  unshadedColor;  // Packed RGBA8 (unlit base color)
    float depth;       // Normalized linear view-space distance.
    float normalX;     // Geometric normal in view space, for merging.
    float normalY;
    float normalZ;
    uint  objectID;
    uint  next;
};

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

// Texture array (set 1, binding 0)
layout(set = 1, binding = 0) uniform sampler2D textures[64];

// ─── Helpers ─────────────────────────────────────────────────────────────────

uint packColor(vec4 color) {
    uint r = uint(clamp(color.r, 0.0, 1.0) * 255.0);
    uint g = uint(clamp(color.g, 0.0, 1.0) * 255.0);
    uint b = uint(clamp(color.b, 0.0, 1.0) * 255.0);
    uint a = uint(clamp(color.a, 0.0, 1.0) * 255.0);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

void main() {
    // Evaluate derivatives before any discard, early return, or divergent branch.
    // The interpolated position lies on the actual rasterized triangle.
    vec3 positionDx = dFdxFine(fragPositionView);
    vec3 positionDy = dFdyFine(fragPositionView);
    vec3 geometricCross = cross(positionDx, positionDy);
    float geometricLength2 = dot(geometricCross, geometricCross);
    float derivativeScale2 = dot(positionDx, positionDx)
        * dot(positionDy, positionDy);

    float shadingLength2 = dot(fragNormalView, fragNormalView);
    bool hasShadingNormal = shadingLength2 > 0.0;
    vec3 shadingNormal = hasShadingNormal
        ? fragNormalView * inversesqrt(shadingLength2)
        : vec3(0.0, 0.0, -1.0);

    // Use a relative degeneracy test so it does not depend on scene units.
    bool hasGeometricNormal = geometricLength2 > 0.0
        && geometricLength2 > 1e-12 * derivativeScale2;
    vec3 mergeNormal = shadingNormal;
    if (hasGeometricNormal) {
        mergeNormal = geometricCross * inversesqrt(geometricLength2);

        // Derivative winding depends on screen orientation. Preserve the mesh's
        // normal orientation, including the distinction between front and back.
        if (hasShadingNormal && dot(mergeNormal, shadingNormal) < 0.0) {
            mergeNormal = -mergeNormal;
        }
        if (!hasShadingNormal) {
            shadingNormal = mergeNormal;
        }
    }

    // ─── Material color ──────────────────────────────────────────────────
    vec3 baseColor = fragColor; // Material Kd baked into vertex color

    // Sample texture if available
    if (pc.textureIndex >= 0) {
        vec4 texColor = texture(textures[pc.textureIndex], fragUV);
        baseColor *= texColor.rgb;

        // Alpha test: discard nearly-transparent pixels (e.g. Sponza leaves)
        if (texColor.a < 0.5) {
            discard;
        }
    }

    // ─── Lighting ────────────────────────────────────────────────────────
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(shadingNormal, lightDir), 0.0) * 0.7 + 0.3;
    vec4 finalColor = vec4(baseColor * diffuse, 1.0);

    // ─── Allocate node ───────────────────────────────────────────────────
    uint nodeIndex = atomicAdd(counter, 1u);
    if (nodeIndex >= nodes.length()) {
        return;
    }

    // Insert at head
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    uint prevHead = imageAtomicExchange(headPointerImage, pixelCoord, nodeIndex);

    // Store view distance directly. Reversing gl_FragCoord.z would amplify
    // nonlinear depth rounding at distance before the merge pass sees it.
    float linDepth = -fragPositionView.z;
    float normalizedDepth = clamp(
        (linDepth - pc.nearPlane) / (pc.farPlane - pc.nearPlane),
        0.0, 1.0
    );

    // Write node
    nodes[nodeIndex].color    = packColor(finalColor);
    nodes[nodeIndex].unshadedColor = packColor(vec4(baseColor, 1.0));
    nodes[nodeIndex].depth    = normalizedDepth;
    nodes[nodeIndex].normalX  = mergeNormal.x;
    nodes[nodeIndex].normalY  = mergeNormal.y;
    nodes[nodeIndex].normalZ  = mergeNormal.z;
    nodes[nodeIndex].objectID = fragObjectID;
    nodes[nodeIndex].next     = prevHead;
}
