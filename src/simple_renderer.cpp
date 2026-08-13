#include "simple_renderer.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "texture_manager.h"
#include "shader_utils.h"
#include "mesh.h"

#include <iostream>
#include <array>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT / DESTROY / RECREATE
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::init(VulkanContext& context, SwapchainManager& swapchain) {
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    if (!createDepthImage(context)) return false;
    if (!createRenderPass(context.device, swapchain.getImageFormat())) return false;
    if (!createFramebuffers(context.device, swapchain)) return false;
    if (!createPipelineLayout(context.device)) return false;
    if (!createPipeline(context.device)) return false;

    std::cout << "[Simple] Initialized: " << m_width << "x" << m_height << std::endl;
    return true;
}

void SimpleRenderer::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);

    if (m_pipeline) vkDestroyPipeline(device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    if (m_renderPass) vkDestroyRenderPass(device, m_renderPass, nullptr);

    destroySizeDependent(device);
}

void SimpleRenderer::recreate(VulkanContext& context, SwapchainManager& swapchain) {
    vkDeviceWaitIdle(context.device);
    destroySizeDependent(context.device);

    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    createDepthImage(context);
    createFramebuffers(context.device, swapchain);
}

void SimpleRenderer::destroySizeDependent(VkDevice device) {
    for (auto fb : m_framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    m_framebuffers.clear();

    if (m_depthView)   { vkDestroyImageView(device, m_depthView, nullptr); m_depthView = VK_NULL_HANDLE; }
    if (m_depthImage)  { vkDestroyImage(device, m_depthImage, nullptr); m_depthImage = VK_NULL_HANDLE; }
    if (m_depthMemory) { vkFreeMemory(device, m_depthMemory, nullptr); m_depthMemory = VK_NULL_HANDLE; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEPTH IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createDepthImage(VulkanContext& context) {
    return createImage(context, m_width, m_height,
                       VK_FORMAT_D32_SFLOAT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT,
                       m_depthImage, m_depthMemory, m_depthView);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDER PASS
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createRenderPass(VkDevice device, VkFormat swapchainFormat) {
    // Attachment 0: Color (swapchain)
    VkAttachmentDescription colorAtt{};
    colorAtt.format = swapchainFormat;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Attachment 1: Depth
    VkAttachmentDescription depthAtt{};
    depthAtt.format = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAtt, depthAtt};

    VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpInfo.attachmentCount = (uint32_t)attachments.size();
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to create render pass" << std::endl;
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAMEBUFFERS
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    m_framebuffers.resize(swapchain.getImageCount());

    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        std::array<VkImageView, 2> attachments = {
            swapchain.getImageView(i),
            m_depthView
        };

        VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbInfo.renderPass = m_renderPass;
        fbInfo.attachmentCount = (uint32_t)attachments.size();
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = m_width;
        fbInfo.height = m_height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_framebuffers[i]) != VK_SUCCESS) {
            std::cerr << "[Simple] Failed to create framebuffer " << i << std::endl;
            return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE LAYOUT
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createPipelineLayout(VkDevice device) {
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SimpleScenePushConstants);

    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &pushRange;

    // If texture manager is available, use its descriptor set layout (set 0)
    VkDescriptorSetLayout texLayout = VK_NULL_HANDLE;
    if (m_textureManager) {
        texLayout = m_textureManager->getDescriptorSetLayout();
        info.setLayoutCount = 1;
        info.pSetLayouts = &texLayout;
    } else {
        info.setLayoutCount = 0;
        info.pSetLayouts = nullptr;
    }

    if (vkCreatePipelineLayout(device, &info, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to create pipeline layout" << std::endl;
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createPipeline(VkDevice device) {
    // ─── Shader Modules ──────────────────────────────────────────────────
    VkShaderModule vertModule = ShaderUtils::createShaderModule(device, "shaders/simple/simple.vert.spv");
    VkShaderModule fragModule = ShaderUtils::createShaderModule(device, "shaders/simple/simple.frag.spv");

    if (!vertModule || !fragModule) {
        std::cerr << "[Simple] Failed to load shaders" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // ─── Vertex Input ────────────────────────────────────────────────────
    VkVertexInputBindingDescription bindingDesc = Vertex::getBindingDescription();
    auto attrDescs = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrDescs.size();
    vertexInput.pVertexAttributeDescriptions = attrDescs.data();

    // ─── Input Assembly ──────────────────────────────────────────────────
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // ─── Viewport & Scissor (dynamic) ───────────────────────────────────
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // ─── Rasterizer ──────────────────────────────────────────────────────
    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // ─── Multisampling ───────────────────────────────────────────────────
    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ─── Depth Stencil ───────────────────────────────────────────────────
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // ─── Color Blending ──────────────────────────────────────────────────
    VkPipelineColorBlendAttachmentState colorBlendAtt{};
    colorBlendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.logicOpEnable = VK_FALSE;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAtt;

    // ─── Create Pipeline ─────────────────────────────────────────────────
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to create graphics pipeline" << std::endl;
        vkDestroyShaderModule(device, vertModule, nullptr);
        vkDestroyShaderModule(device, fragModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    std::cout << "[Simple] Pipeline created." << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECORD COMMANDS
// ═══════════════════════════════════════════════════════════════════════════════

void SimpleRenderer::recordCommands(VkCommandBuffer cmd,
                                    VulkanContext& context,
                                    SwapchainManager& swapchain,
                                    VkBuffer vertexBuffer,
                                    VkBuffer indexBuffer,
                                    const std::vector<SceneObject>& objects,
                                    const glm::mat4& viewProj,
                                    const glm::mat4& view,
                                    const glm::mat4& projection,
                                    uint32_t swapchainImageIndex) {

    // ─── Begin Render Pass ───────────────────────────────────────────────
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { {0.02f, 0.02f, 0.08f, 1.0f} };
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass = m_renderPass;
    rpBegin.framebuffer = m_framebuffers[swapchainImageIndex];
    rpBegin.renderArea = { {0, 0}, {m_width, m_height} };
    rpBegin.clearValueCount = (uint32_t)clearValues.size();
    rpBegin.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // ─── Bind Pipeline ───────────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // ─── Dynamic Viewport (flipped Y for Vulkan) ────────────────────────
    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = (float)m_height;
    vp.width = (float)m_width;
    vp.height = -(float)m_height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{ {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // ─── Bind Texture Descriptor Set (set 0) ────────────────────────────
    if (m_textureManager) {
        VkDescriptorSet texSet = m_textureManager->getDescriptorSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 0, 1, &texSet, 0, nullptr);
    }

    // ─── Bind Vertex & Index Buffers ─────────────────────────────────────
    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // ─── Draw Each Object ────────────────────────────────────────────────
    for (const auto& obj : objects) {
        SimpleScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = viewProj;
        pc.view = view;
        pc.textureIndex = obj.textureIndex;

        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
    }

    // ─── End Render Pass ─────────────────────────────────────────────────
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

bool SimpleRenderer::createImage(VulkanContext& context, uint32_t w, uint32_t h,
                                 VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                 VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = format;
    imgInfo.extent = {w, h, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = usage;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(context.device, &imgInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to create image" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to allocate image memory" << std::endl;
        return false;
    }
    vkBindImageMemory(context.device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        std::cerr << "[Simple] Failed to create image view" << std::endl;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: FIND MEMORY TYPE
// ═══════════════════════════════════════════════════════════════════════════════

uint32_t SimpleRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    std::cerr << "[Simple] No suitable memory type found!" << std::endl;
    return 0;
}