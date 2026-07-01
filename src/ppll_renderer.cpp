#include "ppll_renderer.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "mesh.h"
#include "push_constants.h"

#include <iostream>
#include <array>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT / DESTROY / RECREATE
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::init(VulkanContext& context, SwapchainManager& swapchain) {
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    if (!createDataStructures(context)) return false;
    if (!createLayerImages(context)) return false;
    if (!createRenderPasses(context.device, swapchain.getImageFormat())) return false;
    if (!createFramebuffers(context.device, swapchain)) return false;
    if (!createSampler(context.device)) return false;
    if (!createDescriptorSetLayouts(context.device)) return false;
    if (!createDescriptorPool(context.device)) return false;
    if (!createDescriptorSets(context.device)) return false;
    if (!createPipelineLayouts(context.device)) return false;
    if (!createPipelines(context.device)) return false;

    std::cout << "[PPLL] Initialized: " << m_width << "x" << m_height
              << ", " << NUM_PEEL_LAYERS << " layers, "
              << m_nodeCapacity << " nodes ("
              << (m_nodeCapacity * sizeof(FragmentNode)) / (1024 * 1024) << " MB)"
              << ", grow=" << growIterations << std::endl;
    return true;
}

void PPLLRenderer::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_collectPipeline, nullptr);
    vkDestroyPipeline(device, m_sortPipeline, nullptr);
    vkDestroyPipeline(device, m_findMinPipeline, nullptr);
    vkDestroyPipeline(device, m_seedPipeline, nullptr);
    vkDestroyPipeline(device, m_growPipeline, nullptr);
    vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    vkDestroyPipeline(device, m_compositePipeline, nullptr);

    vkDestroyPipelineLayout(device, m_collectPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, m_fullscreenPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, m_compositePipelineLayout, nullptr);

    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_ppllDescLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_compositeDescLayout, nullptr);

    vkDestroySampler(device, m_sampler, nullptr);

    vkDestroyRenderPass(device, m_noAttachmentRenderPass, nullptr);
    vkDestroyRenderPass(device, m_layerRenderPass, nullptr);
    vkDestroyRenderPass(device, m_compositeRenderPass, nullptr);

    destroySizeDependent(device);
}

void PPLLRenderer::recreate(VulkanContext& context, SwapchainManager& swapchain) {
    vkDeviceWaitIdle(context.device);

    vkDestroyDescriptorPool(context.device, m_descriptorPool, nullptr);
    m_descriptorPool = VK_NULL_HANDLE;

    destroySizeDependent(context.device);

    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    createDataStructures(context);
    createLayerImages(context);
    createFramebuffers(context.device, swapchain);
    createDescriptorPool(context.device);
    createDescriptorSets(context.device);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createDataStructures(VulkanContext& context) {
    // Head pointer image (R32_UINT)
    {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R32_UINT;
        info.extent = {m_width, m_height, 1};
        info.mipLevels = 1; info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(context.device, &info, nullptr, &m_headPointerImage) != VK_SUCCESS) return false;

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(context.device, m_headPointerImage, &memReq);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(context.device, &alloc, nullptr, &m_headPointerMemory) != VK_SUCCESS) return false;
        vkBindImageMemory(context.device, m_headPointerImage, m_headPointerMemory, 0);

        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = m_headPointerImage;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R32_UINT;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(context.device, &view, nullptr, &m_headPointerView) != VK_SUCCESS) return false;
    }

    // Label image (R32_UINT)
    if (!createImage(context, m_width, m_height, VK_FORMAT_R32_UINT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     m_labelImage, m_labelMemory, m_labelView)) return false;

    // Consumed depth image (R32_SFLOAT)
    if (!createImage(context, m_width, m_height, VK_FORMAT_R32_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     m_consumedDepthImage, m_consumedDepthMemory, m_consumedDepthView)) return false;

    // Node buffer
    m_nodeCapacity = m_width * m_height * AVG_FRAGS_PER_PIXEL;
    if (!createBuffer(context, m_nodeCapacity * sizeof(FragmentNode),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_nodeBuffer, m_nodeBufferMemory)) return false;

    // Atomic counter (4 bytes)
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_atomicCounterBuffer, m_atomicCounterMemory)) return false;

    // State buffer (4 bytes: globalMinUint)
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_stateBuffer, m_stateBufferMemory)) return false;

    return true;
}

bool PPLLRenderer::createLayerImages(VulkanContext& context) {
    m_layerImages.resize(NUM_PEEL_LAYERS);
    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
        if (!createImage(context, m_width, m_height, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         m_layerImages[i].image, m_layerImages[i].memory, m_layerImages[i].view))
            return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDER PASSES
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createRenderPasses(VkDevice device, VkFormat swapchainFormat) {
    // No-attachment render pass
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_noAttachmentRenderPass) != VK_SUCCESS) return false;
    }

    // Layer render pass (RGBA8 output)
    {
        VkAttachmentDescription att{};
        att.format = VK_FORMAT_R8G8B8A8_UNORM;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 1; rp.pAttachments = &att;
        rp.subpassCount = 1; rp.pSubpasses = &subpass;
        rp.dependencyCount = 1; rp.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_layerRenderPass) != VK_SUCCESS) return false;
    }

    // Composite render pass (swapchain)
    {
        VkAttachmentDescription att{};
        att.format = swapchainFormat;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 1; rp.pAttachments = &att;
        rp.subpassCount = 1; rp.pSubpasses = &subpass;
        rp.dependencyCount = 1; rp.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_compositeRenderPass) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAMEBUFFERS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    // No-attachment framebuffer
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_noAttachmentRenderPass;
        fb.attachmentCount = 0;
        fb.width = m_width; fb.height = m_height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_noAttachmentFramebuffer) != VK_SUCCESS) return false;
    }

    // Layer framebuffers
    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_layerRenderPass;
        fb.attachmentCount = 1; fb.pAttachments = &m_layerImages[i].view;
        fb.width = m_width; fb.height = m_height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_layerImages[i].framebuffer) != VK_SUCCESS) return false;
    }

    // Composite framebuffers
    m_compositeFramebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        VkImageView view = swapchain.getImageView(i);
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_compositeRenderPass;
        fb.attachmentCount = 1; fb.pAttachments = &view;
        fb.width = swapchain.getExtent().width; fb.height = swapchain.getExtent().height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_compositeFramebuffers[i]) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createSampler(VkDevice device) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = VK_FILTER_NEAREST;
    info.minFilter = VK_FILTER_NEAREST;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.maxLod = 1.0f;
    return vkCreateSampler(device, &info, nullptr, &m_sampler) == VK_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR SET LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createDescriptorSetLayouts(VkDevice device) {
    // PPLL set: 6 bindings
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};  // head
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // nodes
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // counter
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}; // state
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};  // label
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};  // consumed depth

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 6; info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_ppllDescLayout) != VK_SUCCESS) return false;
    }

    // Composite set
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(NUM_PEEL_LAYERS);
        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = NUM_PEEL_LAYERS; info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_compositeDescLayout) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR POOL & SETS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 3> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3};     // head + label + consumed
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};    // nodes + counter + state
    sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NUM_PEEL_LAYERS};

    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount = (uint32_t)sizes.size(); info.pPoolSizes = sizes.data();
    info.maxSets = 2;
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool PPLLRenderer::createDescriptorSets(VkDevice device) {
    // PPLL set
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1; alloc.pSetLayouts = &m_ppllDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_ppllDescSet) != VK_SUCCESS) return false;

        VkDescriptorImageInfo headInfo{};
        headInfo.imageView = m_headPointerView;
        headInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo nodeInfo{m_nodeBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo counterInfo{m_atomicCounterBuffer, 0, sizeof(uint32_t)};
        VkDescriptorBufferInfo stateInfo{m_stateBuffer, 0, sizeof(uint32_t)};

        VkDescriptorImageInfo labelInfo{};
        labelInfo.imageView = m_labelView;
        labelInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo consumedInfo{};
        consumedInfo.imageView = m_consumedDepthView;
        consumedInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 6> writes{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = m_ppllDescSet; writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].descriptorCount = 1; writes[0].pImageInfo = &headInfo;

        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = m_ppllDescSet; writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1; writes[1].pBufferInfo = &nodeInfo;

        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[2].dstSet = m_ppllDescSet; writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1; writes[2].pBufferInfo = &counterInfo;

        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[3].dstSet = m_ppllDescSet; writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1; writes[3].pBufferInfo = &stateInfo;

        writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[4].dstSet = m_ppllDescSet; writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[4].descriptorCount = 1; writes[4].pImageInfo = &labelInfo;

        writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[5].dstSet = m_ppllDescSet; writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[5].descriptorCount = 1; writes[5].pImageInfo = &consumedInfo;

        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Composite set
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1; alloc.pSetLayouts = &m_compositeDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_compositeDescSet) != VK_SUCCESS) return false;

        std::vector<VkDescriptorImageInfo> imgInfos(NUM_PEEL_LAYERS);
        std::vector<VkWriteDescriptorSet> writes(NUM_PEEL_LAYERS);
        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
            imgInfos[i].sampler = m_sampler;
            imgInfos[i].imageView = m_layerImages[i].view;
            imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = m_compositeDescSet; writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1; writes[i].pImageInfo = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, NUM_PEEL_LAYERS, writes.data(), 0, nullptr);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createPipelineLayouts(VkDevice device) {
    // Collect: ScenePushConstants
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(ScenePushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1; info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_collectPipelineLayout) != VK_SUCCESS) return false;
    }

    // Fullscreen: { float depthThreshold, float colorThreshold }
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(float) * 2;

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1; info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_fullscreenPipelineLayout) != VK_SUCCESS) return false;
    }

    // Composite: { uint numLayers, int displayLayer }
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(uint32_t) + sizeof(int32_t);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &m_compositeDescLayout;
        info.pushConstantRangeCount = 1; info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_compositePipelineLayout) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINES
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createPipelines(VkDevice device) {
    std::string sd = "shaders";

    VkShaderModule collectVert    = ShaderUtils::createShaderModule(device, sd + "/ppll_collect.vert.spv");
    VkShaderModule collectFrag    = ShaderUtils::createShaderModule(device, sd + "/ppll_collect.frag.spv");
    VkShaderModule fullscreenVert = ShaderUtils::createShaderModule(device, sd + "/ppll_resolve.vert.spv");
    VkShaderModule sortFrag       = ShaderUtils::createShaderModule(device, sd + "/ppll_sort.frag.spv");
    VkShaderModule findMinFrag    = ShaderUtils::createShaderModule(device, sd + "/ppll_find_min.frag.spv");
    VkShaderModule seedFrag       = ShaderUtils::createShaderModule(device, sd + "/ppll_seed.frag.spv");
    VkShaderModule growFrag       = ShaderUtils::createShaderModule(device, sd + "/ppll_grow.frag.spv");
    VkShaderModule resolveFrag    = ShaderUtils::createShaderModule(device, sd + "/ppll_resolve.frag.spv");
    VkShaderModule compositeFrag  = ShaderUtils::createShaderModule(device, sd + "/ppll_composite.frag.spv");

    if (!collectVert || !collectFrag || !fullscreenVert || !sortFrag ||
        !findMinFrag || !seedFrag || !growFrag || !resolveFrag || !compositeFrag) {
        std::cerr << "[PPLL] Failed to load shaders!" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription vtxBind = Vertex::getBindingDescription();
    auto vtxAttr = Vertex::getAttributeDescriptions();
    PipelineBuilder b;

    // Collect
    m_collectPipeline = b
        .setVertexShader(collectVert).setFragmentShader(collectFrag)
        .setVertexInput({vtxBind}, vtxAttr)
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_collectPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_collectPipeline) return false;

    // Sort
    b.clear();
    m_sortPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(sortFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_sortPipeline) return false;

    // Find-min
    b.clear();
    m_findMinPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(findMinFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_findMinPipeline) return false;

    // Seed
    b.clear();
    m_seedPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(seedFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_seedPipeline) return false;

    // Grow
    b.clear();
    m_growPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(growFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_growPipeline) return false;

    // Resolve
    b.clear();
    m_resolvePipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(resolveFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().disableBlending()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_layerRenderPass, 0).build(device);
    if (!m_resolvePipeline) return false;

    // Composite
    b.clear();
    m_compositePipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(compositeFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().disableBlending()
        .setPipelineLayout(m_compositePipelineLayout)
        .setRenderPass(m_compositeRenderPass, 0).build(device);
    if (!m_compositePipeline) return false;

    vkDestroyShaderModule(device, collectVert, nullptr);
    vkDestroyShaderModule(device, collectFrag, nullptr);
    vkDestroyShaderModule(device, fullscreenVert, nullptr);
    vkDestroyShaderModule(device, sortFrag, nullptr);
    vkDestroyShaderModule(device, findMinFrag, nullptr);
    vkDestroyShaderModule(device, seedFrag, nullptr);
    vkDestroyShaderModule(device, growFrag, nullptr);
    vkDestroyShaderModule(device, resolveFrag, nullptr);
    vkDestroyShaderModule(device, compositeFrag, nullptr);

    std::cout << "[PPLL] 7 pipelines created." << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND RECORDING — MAIN ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordCommands(VkCommandBuffer cmd,
                                   VulkanContext& context,
                                   SwapchainManager& swapchain,
                                   VkBuffer vertexBuffer,
                                   VkBuffer indexBuffer,
                                   const std::vector<SceneObject>& objects,
                                   const glm::mat4& viewProj,
                                   uint32_t swapchainImageIndex) {
    // 1. Clear everything
    recordClearPhase(cmd);

    // 2. Collect
    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj);
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // 3. Sort
    recordSortPhase(cmd);
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    // 4. Iterative region growing peel
    for (uint32_t layer = 0; layer < NUM_PEEL_LAYERS; layer++) {
        // a. Find-min
        recordFindMinPhase(cmd);
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        // b. Seed
        recordSeedPhase(cmd);
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        // c. Grow × K
        recordGrowPhase(cmd);

        // d. Resolve
        recordResolvePhase(cmd, layer);
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);

        // e. Clear labels + reset globalMin for next layer
        recordClearLabelsPhase(cmd);
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // 5. Composite
    recordCompositePhase(cmd, swapchainImageIndex);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: CLEAR
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordClearPhase(VkCommandBuffer cmd) {
    // Transition all storage images to TRANSFER_DST
    std::array<VkImageMemoryBarrier, 3> imgBarriers{};
    for (auto& b : imgBarriers) {
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    imgBarriers[0].image = m_headPointerImage;
    imgBarriers[1].image = m_labelImage;
    imgBarriers[2].image = m_consumedDepthImage;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 3, imgBarriers.data());

    // Clear head → 0xFFFFFFFF
    VkClearColorValue headClear{}; headClear.uint32[0] = PPLL_NULL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_headPointerImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &headClear, 1, &range);

    // Clear label → 0
    VkClearColorValue labelClear{}; labelClear.uint32[0] = 0;
    vkCmdClearColorImage(cmd, m_labelImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &labelClear, 1, &range);

    // Clear consumed depth → 0.0f
    VkClearColorValue depthClear{}; depthClear.float32[0] = 0.0f;
    vkCmdClearColorImage(cmd, m_consumedDepthImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depthClear, 1, &range);

    // Reset buffers
    vkCmdFillBuffer(cmd, m_atomicCounterBuffer, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, m_stateBuffer, 0, sizeof(uint32_t), PPLL_NULL);

    // Transition all images back to GENERAL
    for (auto& b : imgBarriers) {
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkMemoryBarrier memBar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &memBar, 0, nullptr, 3, imgBarriers.data());
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: COLLECT
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordCollectPhase(VkCommandBuffer cmd, VkBuffer vertexBuffer, VkBuffer indexBuffer,
                                       const std::vector<SceneObject>& objects, const glm::mat4& viewProj) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipeline);

    VkViewport vp{0, (float)m_height, (float)m_width, -(float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto& obj : objects) {
        ScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = viewProj;
        pc.objectID = obj.objectID;
        vkCmdPushConstants(cmd, m_collectPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: SORT
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordSortPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_sortPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    float pc[2] = {depthThreshold, colorThreshold};
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: FIND MIN
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordFindMinPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_findMinPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    float pc[2] = {depthThreshold, colorThreshold};
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}
// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: SEED
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordSeedPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_seedPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    float pc[2] = {depthThreshold, colorThreshold};
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: GROW (repeated K times with barriers between iterations)
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordGrowPhase(VkCommandBuffer cmd) {
    float pc[2] = {depthThreshold, colorThreshold};

    for (uint32_t i = 0; i < growIterations; i++) {
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = m_noAttachmentRenderPass;
        rp.framebuffer = m_noAttachmentFramebuffer;
        rp.renderArea = {{0, 0}, {m_width, m_height}};
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_growPipeline);

        VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc = {{0,0}, {m_width, m_height}};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        // Barrier between grow iterations so label writes are visible to next iteration
        if (i < growIterations - 1) {
            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }

    // Final barrier after all grow iterations before resolve reads labels
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: RESOLVE (outputs labeled pixels to layer color, updates consumed depth)
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex) {
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_layerRenderPass;
    rp.framebuffer = m_layerImages[layerIndex].framebuffer;
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resolvePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    float pc[2] = {depthThreshold, colorThreshold};
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: CLEAR LABELS + RESET GLOBAL MIN (between layers)
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordClearLabelsPhase(VkCommandBuffer cmd) {
    // Transition label image to TRANSFER_DST
    VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imgBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.image = m_labelImage;
    imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    // Clear label to 0
    VkClearColorValue labelClear{};
    labelClear.uint32[0] = 0;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_labelImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &labelClear, 1, &range);

    // Reset globalMinUint to 0xFFFFFFFF
    vkCmdFillBuffer(cmd, m_stateBuffer, 0, sizeof(uint32_t), PPLL_NULL);

    // Transition label back to GENERAL
    imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkMemoryBarrier memBar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &memBar, 0, nullptr, 1, &imgBarrier);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE: COMPOSITE (blend all layers → swapchain)
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex) {
    VkClearValue clear{};
    clear.color = {{0.02f, 0.02f, 0.08f, 1.0f}};

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_compositeRenderPass;
    rp.framebuffer = m_compositeFramebuffers[swapchainImageIndex];
    rp.renderArea = {{0, 0}, {m_width, m_height}};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipelineLayout, 0, 1, &m_compositeDescSet, 0, nullptr);

    struct {
        uint32_t numLayers;
        int32_t  displayLayer;
    } compositePC;
    compositePC.numLayers = NUM_PEEL_LAYERS;
    compositePC.displayLayer = debugDisplayLayer;
    vkCmdPushConstants(cmd, m_compositePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(compositePC), &compositePC);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: MEMORY BARRIER
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordMemoryBarrier(VkCommandBuffer cmd,
                                        VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                        VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createImage(VulkanContext& context, uint32_t w, uint32_t h,
                                VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = format;
    imgInfo.extent = {w, h, 1};
    imgInfo.mipLevels = 1; imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = usage;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(context.device, &imgInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create image" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to allocate image memory" << std::endl;
        return false;
    }
    vkBindImageMemory(context.device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspect;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create image view" << std::endl;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createBuffer(VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to allocate buffer memory" << std::endl;
        return false;
    }
    vkBindBufferMemory(context.device, buffer, memory, 0);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: FIND MEMORY TYPE
// ═══════════════════════════════════════════════════════════════════════════════

uint32_t PPLLRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    std::cerr << "[PPLL] No suitable memory type found!" << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESTROY SIZE-DEPENDENT RESOURCES
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::destroySizeDependent(VkDevice device) {
    // Head pointer
    if (m_headPointerView)   { vkDestroyImageView(device, m_headPointerView, nullptr); m_headPointerView = VK_NULL_HANDLE; }
    if (m_headPointerImage)  { vkDestroyImage(device, m_headPointerImage, nullptr); m_headPointerImage = VK_NULL_HANDLE; }
    if (m_headPointerMemory) { vkFreeMemory(device, m_headPointerMemory, nullptr); m_headPointerMemory = VK_NULL_HANDLE; }

    // Label
    if (m_labelView)   { vkDestroyImageView(device, m_labelView, nullptr); m_labelView = VK_NULL_HANDLE; }
    if (m_labelImage)  { vkDestroyImage(device, m_labelImage, nullptr); m_labelImage = VK_NULL_HANDLE; }
    if (m_labelMemory) { vkFreeMemory(device, m_labelMemory, nullptr); m_labelMemory = VK_NULL_HANDLE; }

    // Consumed depth
    if (m_consumedDepthView)   { vkDestroyImageView(device, m_consumedDepthView, nullptr); m_consumedDepthView = VK_NULL_HANDLE; }
    if (m_consumedDepthImage)  { vkDestroyImage(device, m_consumedDepthImage, nullptr); m_consumedDepthImage = VK_NULL_HANDLE; }
    if (m_consumedDepthMemory) { vkFreeMemory(device, m_consumedDepthMemory, nullptr); m_consumedDepthMemory = VK_NULL_HANDLE; }

    // Node buffer
    if (m_nodeBuffer)       { vkDestroyBuffer(device, m_nodeBuffer, nullptr); m_nodeBuffer = VK_NULL_HANDLE; }
    if (m_nodeBufferMemory) { vkFreeMemory(device, m_nodeBufferMemory, nullptr); m_nodeBufferMemory = VK_NULL_HANDLE; }
    m_nodeCapacity = 0;

    // Atomic counter
    if (m_atomicCounterBuffer) { vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr); m_atomicCounterBuffer = VK_NULL_HANDLE; }
    if (m_atomicCounterMemory) { vkFreeMemory(device, m_atomicCounterMemory, nullptr); m_atomicCounterMemory = VK_NULL_HANDLE; }

    // State buffer
    if (m_stateBuffer)       { vkDestroyBuffer(device, m_stateBuffer, nullptr); m_stateBuffer = VK_NULL_HANDLE; }
    if (m_stateBufferMemory) { vkFreeMemory(device, m_stateBufferMemory, nullptr); m_stateBufferMemory = VK_NULL_HANDLE; }

    // Layer images
    for (auto& layer : m_layerImages) {
        if (layer.framebuffer) { vkDestroyFramebuffer(device, layer.framebuffer, nullptr); layer.framebuffer = VK_NULL_HANDLE; }
        if (layer.view)        { vkDestroyImageView(device, layer.view, nullptr); layer.view = VK_NULL_HANDLE; }
        if (layer.image)       { vkDestroyImage(device, layer.image, nullptr); layer.image = VK_NULL_HANDLE; }
        if (layer.memory)      { vkFreeMemory(device, layer.memory, nullptr); layer.memory = VK_NULL_HANDLE; }
    }
    m_layerImages.clear();

    // No-attachment framebuffer
    if (m_noAttachmentFramebuffer) { vkDestroyFramebuffer(device, m_noAttachmentFramebuffer, nullptr); m_noAttachmentFramebuffer = VK_NULL_HANDLE; }

    // Composite framebuffers
    for (auto fb : m_compositeFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    m_compositeFramebuffers.clear();
}