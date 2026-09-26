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

const uint UF_NULL   = 0xFFFFFFFFu;
const uint MAX_FRAGS = 64u;

void main()
{
    ivec2 pix = ivec2(gl_FragCoord.xy);
    if (pix.x >= int(pc.screenWidth) || pix.y >= int(pc.screenHeight)) discard;

    uint head = imageLoad(headPointerImage, pix).r;
    if (head == UF_NULL) discard;

    // Gather node indices in list order.
    uint idx[MAX_FRAGS];
    uint count = 0u;

    uint cur = head;
    while (cur != UF_NULL && count < MAX_FRAGS)
    {
        idx[count] = cur;
        cur = nodes[cur].next;
        count++;
    }

    if (count < 2u)
    {
        discard; // nothing to sort
    }

    // Insertion sort on the indices by ascending depth.
    for (uint i = 1u; i < count; ++i)
    {
        uint keyIdx = idx[i];
        float keyDepth = nodes[keyIdx].depth;
        uint j = i;
        while (j > 0u && nodes[idx[j - 1u]].depth > keyDepth)
        {
            idx[j] = idx[j - 1u];
            j--;
        }
        idx[j] = keyIdx;
    }

    // Rewrite the linked list "next" pointers so head = idx[0], last.next = NULL.
    imageStore(headPointerImage, pix, uvec4(idx[0], 0, 0, 0));

    for (uint i = 0u; i + 1u < count; ++i)
    {
        nodes[idx[i]].next = idx[i + 1u];
    }
    nodes[idx[count - 1u]].next = UF_NULL;

    discard; // No color output — this is a "compute-in-fragment" pass.
}