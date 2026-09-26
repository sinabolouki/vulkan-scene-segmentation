#version 450

struct FragmentNode {
    uint color;
    uint unshadedColor;
    float depth;
    float normalX;
    float normalY;
    float normalZ;
    uint objectID;
    uint next;
};

layout(set = 0, binding = 0, r32ui) uniform uimage2D headPointerImage;

layout(set = 0, binding = 1, std430) buffer NodeBuffer {
    FragmentNode nodes[];
};

layout(set = 0, binding = 2, std430) buffer AtomicCounter {
    uint counter;
};

layout(push_constant) uniform Push {
    float planeThreshold;
    float proximityScale;
    uint  screenWidth;
    uint  screenHeight;
    uint  targetLayer;
    float nearPlane;
    float farPlane;
    float projX;
    float projY;
    float colorThreshold;
    uint  mergeMode;
} pc;

layout(location = 0) out vec4 outColor;

const uint UF_NULL = 0xFFFFFFFFu;

vec4 unpackRGBA8(uint p)
{
    return vec4(
        float((p      ) & 0xFFu),
        float((p >>  8) & 0xFFu),
        float((p >> 16) & 0xFFu),
        float((p >> 24) & 0xFFu)) / 255.0;
}

void main()
{
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint head = imageLoad(headPointerImage, pix).r;

    if (head == UF_NULL)
    {
        outColor = vec4(0.0);
        return;
    }

    // Walk `targetLayer` steps.
    uint cur = head;
    for (uint i = 0u; i < pc.targetLayer; ++i)
    {
        if (cur == UF_NULL) break;
        cur = nodes[cur].next;
    }

    if (cur == UF_NULL)
    {
        outColor = vec4(0.0);
        return;
    }

    outColor = unpackRGBA8(nodes[cur].color);
}