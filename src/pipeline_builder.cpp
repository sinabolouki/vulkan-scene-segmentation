#include "pipeline_builder.h"
#include <iostream>
#include <cassert>

// Shader Stages
PipelineBuilder& PipelineBuilder::addShaderStage(VkShaderStageFlagBits stage,
                                                  VkShaderModule module,
                                                  const char* entryPoint) {
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = stage;
    stageInfo.module = module;
    stageInfo.pName = entryPoint;
    m_shaderStages.push_back(stageInfo);
    return *this;
}

PipelineBuilder& PipelineBuilder::setVertexShader(VkShaderModule module) {
    return addShaderStage(VK_SHADER_STAGE_VERTEX_BIT, module);
}

PipelineBuilder& PipelineBuilder::setFragmentShader(VkShaderModule module) {
    return addShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, module);
}

// Vertex Input
PipelineBuilder& PipelineBuilder::setVertexInput(
    const std::vector<VkVertexInputBindingDescription>& bindings,
    const std::vector<VkVertexInputAttributeDescription>& attributes) {
    m_vertexBindings = bindings;
    m_vertexAttributes = attributes;
    return *this;
}

PipelineBuilder& PipelineBuilder::setNoVertexInput() {
    m_vertexBindings.clear();
    m_vertexAttributes.clear();
    return *this;
}

// Input Assembly
PipelineBuilder& PipelineBuilder::setTopology(VkPrimitiveTopology topology) {
    m_topology = topology;
    return *this;
}

// Viewport / Scissor
PipelineBuilder& PipelineBuilder::setViewport(float x, float y, float width, float height,
                                               float minDepth, float maxDepth) {
    m_viewport.x = x;
    m_viewport.y = y;
    m_viewport.width = width;
    m_viewport.height = height;
    m_viewport.minDepth = minDepth;
    m_viewport.maxDepth = maxDepth;
    m_useDynamicViewport = false;
    return *this;
}

PipelineBuilder& PipelineBuilder::setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    m_scissor.offset = {x, y};
    m_scissor.extent = {width, height};
    m_useDynamicViewport = false;
    return *this;
}

PipelineBuilder& PipelineBuilder::setDynamicViewportAndScissor() {
    m_useDynamicViewport = true;
    return *this;
}

// Rasterization
PipelineBuilder& PipelineBuilder::setPolygonMode(VkPolygonMode mode) {
    m_polygonMode = mode;
    return *this;
}

PipelineBuilder& PipelineBuilder::setCullMode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
    m_cullMode = cullMode;
    m_frontFace = frontFace;
    return *this;
}

PipelineBuilder& PipelineBuilder::setLineWidth(float width) {
    m_lineWidth = width;
    return *this;
}

PipelineBuilder& PipelineBuilder::enableDepthBias(float constantFactor, float clamp, float slopeFactor) {
    m_depthBiasEnable = true;
    m_depthBiasConstant = constantFactor;
    m_depthBiasClamp = clamp;
    m_depthBiasSlope = slopeFactor;
    return *this;
}

PipelineBuilder& PipelineBuilder::disableDepthBias() {
    m_depthBiasEnable = false;
    return *this;
}

// Multisampling
PipelineBuilder& PipelineBuilder::setMultisampling(VkSampleCountFlagBits samples) {
    m_sampleCount = samples;
    return *this;
}

// Depth / Stencil
PipelineBuilder& PipelineBuilder::enableDepthTest(bool write, VkCompareOp compareOp) {
    m_depthTestEnable = true;
    m_depthWriteEnable = write;
    m_depthCompareOp = compareOp;
    return *this;
}

PipelineBuilder& PipelineBuilder::disableDepthTest() {
    m_depthTestEnable = false;
    m_depthWriteEnable = false;
    return *this;
}

PipelineBuilder& PipelineBuilder::enableDepthBounds(float min, float max) {
    m_depthBoundsEnable = true;
    m_minDepthBounds = min;
    m_maxDepthBounds = max;
    return *this;
}

// Color Blending
PipelineBuilder& PipelineBuilder::setColorBlendAttachment(VkPipelineColorBlendAttachmentState state) {
    if (m_colorBlendAttachments.empty()) {
        m_colorBlendAttachments.push_back(state);
    } else {
        m_colorBlendAttachments[0] = state;
    }
    return *this;
}

PipelineBuilder& PipelineBuilder::enableAlphaBlending() {
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    if (m_colorBlendAttachments.empty()) {
        m_colorBlendAttachments.push_back(blend);
    } else {
        m_colorBlendAttachments[0] = blend;
    }
    return *this;
}

PipelineBuilder& PipelineBuilder::disableBlending() {
    VkPipelineColorBlendAttachmentState noBlend{};
    noBlend.blendEnable = VK_FALSE;
    noBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    if (m_colorBlendAttachments.empty()) {
        m_colorBlendAttachments.push_back(noBlend);
    } else {
        m_colorBlendAttachments[0] = noBlend;
    }
    return *this;
}

PipelineBuilder& PipelineBuilder::setColorBlendAttachmentCount(uint32_t count) {
    // Use the first attachment state as template for all
    VkPipelineColorBlendAttachmentState base{};
    if (!m_colorBlendAttachments.empty()) {
        base = m_colorBlendAttachments[0];
    } else {
        base.blendEnable = VK_FALSE;
        base.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    m_colorBlendAttachments.resize(count, base);
    return *this;
}

PipelineBuilder& PipelineBuilder::setNoColorAttachments() {
    m_colorBlendAttachments.clear();
    return *this;
}

// Dynamic State
PipelineBuilder& PipelineBuilder::setDynamicStates(const std::vector<VkDynamicState>& states) {
    m_dynamicStates = states;
    return *this;
}

// Layout & Render Pass
PipelineBuilder& PipelineBuilder::setPipelineLayout(VkPipelineLayout layout) {
    m_pipelineLayout = layout;
    return *this;
}

PipelineBuilder& PipelineBuilder::setRenderPass(VkRenderPass renderPass, uint32_t subpass) {
    m_renderPass = renderPass;
    m_subpass = subpass;
    return *this;
}

// Build
VkPipeline PipelineBuilder::build(VkDevice device) {
    assert(m_pipelineLayout != VK_NULL_HANDLE && "Pipeline layout must be set");
    assert(m_renderPass != VK_NULL_HANDLE && "Render pass must be set");
    assert(!m_shaderStages.empty() && "At least one shader stage required");

    // ─── Vertex Input ───
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(m_vertexBindings.size());
    vertexInputInfo.pVertexBindingDescriptions = m_vertexBindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_vertexAttributes.size());
    vertexInputInfo.pVertexAttributeDescriptions = m_vertexAttributes.data();

    // ─── Input Assembly ───
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = m_topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // ─── Viewport / Scissor ───
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    if (!m_useDynamicViewport) {
        viewportState.pViewports = &m_viewport;
        viewportState.pScissors = &m_scissor;
    }
    // If dynamic, viewport/scissor are set via vkCmdSetViewport/vkCmdSetScissor

    // ─── Rasterization ───
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = m_polygonMode;
    rasterizer.cullMode = m_cullMode;
    rasterizer.frontFace = m_frontFace;
    rasterizer.lineWidth = m_lineWidth;
    rasterizer.depthBiasEnable = m_depthBiasEnable ? VK_TRUE : VK_FALSE;
    rasterizer.depthBiasConstantFactor = m_depthBiasConstant;
    rasterizer.depthBiasClamp = m_depthBiasClamp;
    rasterizer.depthBiasSlopeFactor = m_depthBiasSlope;

    // ─── Multisampling ───
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = m_sampleCount;
    multisampling.sampleShadingEnable = VK_FALSE;

    // ─── Depth / Stencil ───
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = m_depthTestEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = m_depthWriteEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = m_depthCompareOp;
    depthStencil.depthBoundsTestEnable = m_depthBoundsEnable ? VK_TRUE : VK_FALSE;
    depthStencil.minDepthBounds = m_minDepthBounds;
    depthStencil.maxDepthBounds = m_maxDepthBounds;
    depthStencil.stencilTestEnable = VK_FALSE;

    // // ─── Color Blending ───
    // // Ensure at least one attachment
    // if (m_colorBlendAttachments.empty()) {
    //     disableBlending();
    // }



    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = static_cast<uint32_t>(m_colorBlendAttachments.size());
    colorBlending.pAttachments = m_colorBlendAttachments.empty() ? nullptr : m_colorBlendAttachments.data();

    // ─── Dynamic State ───
    // Add viewport/scissor as dynamic if using dynamic viewport
    std::vector<VkDynamicState> dynamicStates = m_dynamicStates;
    if (m_useDynamicViewport) {
        // Check if already included
        bool hasViewport = false, hasScissor = false;
        for (auto s : dynamicStates) {
            if (s == VK_DYNAMIC_STATE_VIEWPORT) hasViewport = true;
            if (s == VK_DYNAMIC_STATE_SCISSOR) hasScissor = true;
        }
        if (!hasViewport) dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
        if (!hasScissor) dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    }

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    // ─── Pipeline Creation ───
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(m_shaderStages.size());
    pipelineInfo.pStages = m_shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = dynamicStates.empty() ? nullptr : &dynamicStateInfo;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.subpass = m_subpass;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                                 &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline! VkResult: " << result << std::endl;
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

// Clear / Reset
void PipelineBuilder::clear() {
    m_shaderStages.clear();
    m_vertexBindings.clear();
    m_vertexAttributes.clear();
    m_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    m_useDynamicViewport = true;
    m_viewport = {};
    m_scissor = {};
    m_polygonMode = VK_POLYGON_MODE_FILL;
    m_cullMode = VK_CULL_MODE_BACK_BIT;
    m_frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    m_lineWidth = 1.0f;
    m_depthBiasEnable = false;
    m_depthBiasConstant = 0.0f;
    m_depthBiasClamp = 0.0f;
    m_depthBiasSlope = 0.0f;
    m_sampleCount = VK_SAMPLE_COUNT_1_BIT;
    m_depthTestEnable = true;
    m_depthWriteEnable = true;
    m_depthCompareOp = VK_COMPARE_OP_LESS;
    m_depthBoundsEnable = false;
    m_minDepthBounds = 0.0f;
    m_maxDepthBounds = 1.0f;
    m_colorBlendAttachments.clear();
    m_dynamicStates.clear();
    m_pipelineLayout = VK_NULL_HANDLE;
    m_renderPass = VK_NULL_HANDLE;
    m_subpass = 0;
}