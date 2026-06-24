#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

class PipelineBuilder {
public:
    // ─── Shader Stages ───
    PipelineBuilder& addShaderStage(VkShaderStageFlagBits stage, VkShaderModule module,
                                    const char* entryPoint = "main");
    PipelineBuilder& setVertexShader(VkShaderModule module);
    PipelineBuilder& setFragmentShader(VkShaderModule module);

    // ─── Vertex Input ───
    PipelineBuilder& setVertexInput(
        const std::vector<VkVertexInputBindingDescription>& bindings,
        const std::vector<VkVertexInputAttributeDescription>& attributes);
    PipelineBuilder& setNoVertexInput(); // For fullscreen quads generated in vertex shader

    // ─── Input Assembly ───
    PipelineBuilder& setTopology(VkPrimitiveTopology topology);

    // ─── Viewport / Scissor (dynamic by default) ───
    PipelineBuilder& setViewport(float x, float y, float width, float height,
                                 float minDepth = 0.0f, float maxDepth = 1.0f);
    PipelineBuilder& setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);
    PipelineBuilder& setDynamicViewportAndScissor(); // Default

    // ─── Rasterization ───
    PipelineBuilder& setPolygonMode(VkPolygonMode mode);
    PipelineBuilder& setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace);
    PipelineBuilder& setLineWidth(float width);
    PipelineBuilder& enableDepthBias(float constantFactor, float clamp, float slopeFactor);
    PipelineBuilder& disableDepthBias();

    // ─── Multisampling ───
    PipelineBuilder& setMultisampling(VkSampleCountFlagBits samples);

    // ─── Depth / Stencil ───
    PipelineBuilder& enableDepthTest(bool write, VkCompareOp compareOp);
    PipelineBuilder& disableDepthTest();
    PipelineBuilder& enableDepthBounds(float min, float max);

    // ─── Color Blending ───
    PipelineBuilder& setColorBlendAttachment(VkPipelineColorBlendAttachmentState state);
    PipelineBuilder& enableAlphaBlending();   // Standard src-alpha, one-minus-src-alpha
    PipelineBuilder& disableBlending();       // Default: write all, no blend
    PipelineBuilder& setColorBlendAttachmentCount(uint32_t count);

    // ─── Dynamic State ───
    PipelineBuilder& setDynamicStates(const std::vector<VkDynamicState>& states);

    // ─── Layout & Render Pass ───
    PipelineBuilder& setPipelineLayout(VkPipelineLayout layout);
    PipelineBuilder& setRenderPass(VkRenderPass renderPass, uint32_t subpass = 0);

    // ─── Build ───
    VkPipeline build(VkDevice device);

    // ─── Reset for reuse ───
    void clear();

private:
    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> m_shaderStages;

    // Vertex input
    std::vector<VkVertexInputBindingDescription> m_vertexBindings;
    std::vector<VkVertexInputAttributeDescription> m_vertexAttributes;

    // Fixed function state
    VkPrimitiveTopology m_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport (used only if not dynamic)
    VkViewport m_viewport{};
    VkRect2D m_scissor{};
    bool m_useDynamicViewport = true;

    // Rasterization
    VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags m_cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace m_frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    float m_lineWidth = 1.0f;
    bool m_depthBiasEnable = false;
    float m_depthBiasConstant = 0.0f;
    float m_depthBiasClamp = 0.0f;
    float m_depthBiasSlope = 0.0f;

    // Multisampling
    VkSampleCountFlagBits m_sampleCount = VK_SAMPLE_COUNT_1_BIT;

    // Depth/Stencil
    bool m_depthTestEnable = true;
    bool m_depthWriteEnable = true;
    VkCompareOp m_depthCompareOp = VK_COMPARE_OP_LESS;
    bool m_depthBoundsEnable = false;
    float m_minDepthBounds = 0.0f;
    float m_maxDepthBounds = 1.0f;

    // Color blend
    std::vector<VkPipelineColorBlendAttachmentState> m_colorBlendAttachments;

    // Dynamic state
    std::vector<VkDynamicState> m_dynamicStates;

    // Layout & render pass
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    uint32_t m_subpass = 0;
};