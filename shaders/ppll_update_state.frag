#version 450

// Runs once per iteration (single pixel is enough)
// Updates consumedDepth = globalMinDepth + threshold

layout(set = 0, binding = 3, std430) buffer StateBuffer {
    uint globalMinUint;
    uint consumedDepthBits;
};

layout(push_constant) uniform PushConstants {
    float depthThreshold;
} pc;

void main() {
    // Only one invocation needs to do this
    if (gl_FragCoord.x < 1.0 && gl_FragCoord.y < 1.0) {
        float globalMin = uintBitsToFloat(globalMinUint);
        float newConsumed = globalMin + pc.depthThreshold;
        consumedDepthBits = floatBitsToUint(newConsumed);

        // Reset globalMinUint for next iteration
        globalMinUint = 0xFFFFFFFFu;
    }
}