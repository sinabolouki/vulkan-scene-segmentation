#version 450
#extension GL_EXT_nonuniform_qualifier : require

// ---------------- Fragment node layout ----------------
// Must match UFFragmentNode in C++ (32 bytes).
struct FragmentNode {
    uint color;         // packed RGBA8 (shaded/textured)
    uint unshadedColor; // packed RGBA8 (raw)
    float depth;        // linear [0,1]
    float normalX;
    float normalY;
    float normalZ;
    uint objectID;
    uint next;          // 0xFFFFFFFF = null
};

// ---------------- Descriptors ----------------
layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

// Bindless textures (matches TextureManager)
layout(set = 1, binding = 0) uniform sampler2D textures[64];

// ---------------- Push constants ----------------
layout(push_constant) uniform Push {
    mat4 model;
    mat4 viewProj;
    mat4 view;
    uint objectID;
    float nearPlane;
    float farPlane;
    int  textureIndex;
} pc;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vNormalView;
layout(location = 3) in float vDepthLinear;

const uint UF_NULL = 0xFFFFFFFFu;

uint packRGBA8(vec4 c)
{
    c = clamp(c, 0.0, 1.0);
    uvec4 u = uvec4(c * 255.0 + 0.5);
    return (u.a << 24) | (u.b << 16) | (u.g << 8) | u.r;
}

void main()
{
    // ---- 1) Determine surface color ------------------------------------
    vec3 baseColor = vColor;
    if (pc.textureIndex >= 0)
    {
        vec4 tex = texture(textures[nonuniformEXT(pc.textureIndex)], vUV);
        baseColor = tex.rgb * vColor;
    }

    // Cheap fake shading using view-space normal (dot with view dir = normal.z).
    // Enough to make layer thumbnails readable for the user.
    float ndotv = max(dot(normalize(vNormalView), vec3(0.0, 0.0, 1.0)), 0.0);
    float ambient = 0.25;
    vec3 shaded = baseColor * (ambient + (1.0 - ambient) * ndotv);

    uint packedShaded   = packRGBA8(vec4(shaded, 1.0));
    uint packedUnshaded = packRGBA8(vec4(baseColor, 1.0));

    // ---- 2) Allocate a new node ---------------------------------------
    uint newIndex = atomicAdd(counter, 1u);
    if (newIndex >= nodes.length())
    {
        // Overflow: drop this fragment.
        discard;
    }

    // ---- 3) Insert into per-pixel linked list -------------------------
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint oldHead = imageAtomicExchange(headPointerImage, pix, newIndex);

    FragmentNode n;
    n.color         = packedShaded;
    n.unshadedColor = packedUnshaded;
    n.depth         = vDepthLinear;
    n.normalX       = vNormalView.x;
    n.normalY       = vNormalView.y;
    n.normalZ       = vNormalView.z;
    n.objectID      = pc.objectID;
    n.next          = oldHead;

    nodes[newIndex] = n;
}