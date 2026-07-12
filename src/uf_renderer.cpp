#include "uf_renderer.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "mesh.h"
#include "uf_push_constants.h"

#include <iostream>
#include <array>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT / DESTROY / RECREATE
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::init(VulkanContext& context, SwapchainManager& swapchain) {
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

    if (enableTiming) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = MAX_TIMESTAMPS;
        vkCreateQueryPool(context.device, &qpInfo, nullptr, &m_timestampQueryPool);

        // Get timestamp period (nanoseconds per tick)
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(context.physical_device, &props);
        m_timestampPeriod = props.limits.timestampPeriod;
    }

    std::cout << "[UF] Initialized: " << m_width << "x" << m_height
              << ", " << NUM_LAYERS << " layers, "
              << m_nodeCapacity << " nodes ("
              << (m_nodeCapacity * sizeof(UFFragmentNode)) / (1024 * 1024) << " MB node buf, "
              << (m_nodeCapacity * sizeof(uint32_t)) / (1024 * 1024) << " MB parent buf)"
              << std::endl;
    return true;
}

void UFRenderer::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_collectPipeline, nullptr);
    vkDestroyPipeline(device, m_sortPipeline, nullptr);
    vkDestroyPipeline(device, m_initParentsPipeline, nullptr);
    vkDestroyPipeline(device, m_mergePipeline, nullptr);
    vkDestroyPipeline(device, m_compressPipeline, nullptr);
    vkDestroyPipeline(device, m_registerRootsPipeline, nullptr);
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

    if (m_timestampQueryPool) vkDestroyQueryPool(device, m_timestampQueryPool, nullptr);

    destroySizeDependent(device);
}

void UFRenderer::recreate(VulkanContext& context, SwapchainManager& swapchain) {
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

bool UFRenderer::createDataStructures(VulkanContext& context) {
    // Head pointer image (R32_UINT)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
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
        alloc.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(context.device, &alloc, nullptr, &m_headPointerMemory) != VK_SUCCESS) return false;
        vkBindImageMemory(context.device, m_headPointerImage, m_headPointerMemory, 0);

        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = m_headPointerImage;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R32_UINT;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(context.device, &view, nullptr, &m_headPointerView) != VK_SUCCESS) return false;
    }

    // Node buffer (SSBO)
    m_nodeCapacity = m_width * m_height * AVG_FRAGS_PER_PIXEL;
    if (!createBuffer(context, m_nodeCapacity * sizeof(UFFragmentNode),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_nodeBuffer, m_nodeBufferMemory)) return false;

    // Atomic counter (4 bytes)
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_atomicCounterBuffer, m_atomicCounterMemory)) return false;

    // Parent buffer (one uint per node)
    if (!createBuffer(context, m_nodeCapacity * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_parentBuffer, m_parentBufferMemory)) return false;

    // Root table (8 entries × 2 uints = 64 bytes)
    if (!createBuffer(context, MAX_GLOBAL_ROOTS * 2 * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_rootTableBuffer, m_rootTableBufferMemory)) return false;

    return true;
}

bool UFRenderer::createLayerImages(VulkanContext& context) {
    m_layerImages.resize(NUM_LAYERS);
    for (uint32_t i = 0; i < NUM_LAYERS; i++) {
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

bool UFRenderer::createRenderPasses(VkDevice device, VkFormat swapchainFormat) {
    // No-attachment render pass (for collect, sort, init, merge, compress)
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_noAttachmentRenderPass) != VK_SUCCESS) return false;
    }

    // Layer render pass (RGBA8 output, used by resolve)
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

    // Composite render pass (swapchain output)
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

bool UFRenderer::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    // No-attachment framebuffer
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_noAttachmentRenderPass;
        fb.attachmentCount = 0;
        fb.width = m_width; fb.height = m_height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_noAttachmentFramebuffer) != VK_SUCCESS) return false;
    }

    // Layer framebuffers (one per layer image)
    for (uint32_t i = 0; i < NUM_LAYERS; i++) {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_layerRenderPass;
        fb.attachmentCount = 1; fb.pAttachments = &m_layerImages[i].view;
        fb.width = m_width; fb.height = m_height; fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_layerImages[i].framebuffer) != VK_SUCCESS) return false;
    }

    // Composite framebuffers (one per swapchain image)
    m_compositeFramebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        VkImageView view = swapchain.getImageView(i);
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_compositeRenderPass;
        fb.attachmentCount = 1; fb.pAttachments = &view;
        fb.width = swapchain.getExtent().width;
        fb.height = swapchain.getExtent().height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_compositeFramebuffers[i]) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createSampler(VkDevice device) {
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

bool UFRenderer::createDescriptorSetLayouts(VkDevice device) {
    // PPLL set: 5 bindings
    {
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = (uint32_t)bindings.size();
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_ppllDescLayout) != VK_SUCCESS) return false;
    }

    // Composite set: NUM_LAYERS combined image samplers
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(NUM_LAYERS);
        for (uint32_t i = 0; i < NUM_LAYERS; i++) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        }

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = NUM_LAYERS;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_compositeDescLayout) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR POOL & SETS
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 3> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};  // nodes + counter + parent + rootTable
    sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NUM_LAYERS};

    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount = (uint32_t)sizes.size();
    info.pPoolSizes = sizes.data();
    info.maxSets = 2;
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool UFRenderer::createDescriptorSets(VkDevice device) {
    // PPLL descriptor set
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &m_ppllDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_ppllDescSet) != VK_SUCCESS) return false;

        VkDescriptorImageInfo headInfo{};
        headInfo.imageView = m_headPointerView;
        headInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo nodeInfo{m_nodeBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo counterInfo{m_atomicCounterBuffer, 0, sizeof(uint32_t)};
        VkDescriptorBufferInfo parentInfo{m_parentBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo rootTableInfo{m_rootTableBuffer, 0, VK_WHOLE_SIZE};

        std::array<VkWriteDescriptorSet, 5> writes{};

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
        writes[3].descriptorCount = 1; writes[3].pBufferInfo = &parentInfo;

        writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[4].dstSet = m_ppllDescSet; writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1; writes[4].pBufferInfo = &rootTableInfo;

        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Composite descriptor set (unchanged)
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &m_compositeDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_compositeDescSet) != VK_SUCCESS) return false;

        std::vector<VkDescriptorImageInfo> imgInfos(NUM_LAYERS);
        std::vector<VkWriteDescriptorSet> writes(NUM_LAYERS);
        for (uint32_t i = 0; i < NUM_LAYERS; i++) {
            imgInfos[i].sampler = m_sampler;
            imgInfos[i].imageView = m_layerImages[i].view;
            imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = m_compositeDescSet; writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1; writes[i].pImageInfo = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, NUM_LAYERS, writes.data(), 0, nullptr);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createPipelineLayouts(VkDevice device) {
    // Collect pipeline layout: push = UFScenePushConstants
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFScenePushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1; info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_collectPipelineLayout) != VK_SUCCESS) return false;
    }

    // Fullscreen pipeline layout: push = UFFullscreenPushConstants
    // Used by sort, init_parents, merge, compress, resolve
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFFullscreenPushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1; info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_fullscreenPipelineLayout) != VK_SUCCESS) return false;
    }

    // Composite pipeline layout: push = UFCompositePushConstants
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFCompositePushConstants);

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

bool UFRenderer::createPipelines(VkDevice device) {
    std::string sd = "shaders/union_find";

    VkShaderModule collectVert      = ShaderUtils::createShaderModule(device, sd + "/uf_collect.vert.spv");
    VkShaderModule collectFrag      = ShaderUtils::createShaderModule(device, sd + "/uf_collect.frag.spv");
    VkShaderModule fullscreenVert   = ShaderUtils::createShaderModule(device, sd + "/uf_fullscreen.vert.spv");
    VkShaderModule sortFrag         = ShaderUtils::createShaderModule(device, sd + "/uf_sort.frag.spv");
    VkShaderModule initFrag         = ShaderUtils::createShaderModule(device, sd + "/uf_init_parents.frag.spv");
    VkShaderModule mergeFrag        = ShaderUtils::createShaderModule(device, sd + "/uf_merge.frag.spv");
    VkShaderModule compressFrag     = ShaderUtils::createShaderModule(device, sd + "/uf_compress.frag.spv");
    VkShaderModule registerFrag     = ShaderUtils::createShaderModule(device, sd + "/uf_register_roots.frag.spv");
    VkShaderModule resolveFrag      = ShaderUtils::createShaderModule(device, sd + "/uf_resolve.frag.spv");
    VkShaderModule compositeFrag    = ShaderUtils::createShaderModule(device, sd + "/uf_composite.frag.spv");

    if (!collectVert || !collectFrag || !fullscreenVert || !sortFrag ||
        !initFrag || !mergeFrag || !compressFrag || !registerFrag || !resolveFrag || !compositeFrag) {
        std::cerr << "[UF] Failed to load one or more shaders!" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription vtxBind = Vertex::getBindingDescription();
    auto vtxAttr = Vertex::getAttributeDescriptions();
    PipelineBuilder b;

    // ─── Collect ───
    m_collectPipeline = b
        .setVertexShader(collectVert).setFragmentShader(collectFrag)
        .setVertexInput({vtxBind}, vtxAttr)
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_collectPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_collectPipeline) return false;

    // ─── Sort ───
    b.clear();
    m_sortPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(sortFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_sortPipeline) return false;

    // ─── Init Parents ───
    b.clear();
    m_initParentsPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(initFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_initParentsPipeline) return false;

    // ─── Merge ───
    b.clear();
    m_mergePipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(mergeFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_mergePipeline) return false;

    // ─── Compress ───
    b.clear();
    m_compressPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(compressFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_compressPipeline) return false;

    // ─── Register Roots ───
    b.clear();
    m_registerRootsPipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(registerFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_registerRootsPipeline) return false;

    // ─── Resolve ───
    b.clear();
    m_resolvePipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(resolveFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().disableBlending()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_layerRenderPass, 0).build(device);
    if (!m_resolvePipeline) return false;

    // ─── Composite ───
    b.clear();
    m_compositePipeline = b
        .setVertexShader(fullscreenVert).setFragmentShader(compositeFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().disableBlending()
        .setPipelineLayout(m_compositePipelineLayout)
        .setRenderPass(m_compositeRenderPass, 0).build(device);
    if (!m_compositePipeline) return false;

    // Cleanup shader modules
    vkDestroyShaderModule(device, collectVert, nullptr);
    vkDestroyShaderModule(device, collectFrag, nullptr);
    vkDestroyShaderModule(device, fullscreenVert, nullptr);
    vkDestroyShaderModule(device, sortFrag, nullptr);
    vkDestroyShaderModule(device, initFrag, nullptr);
    vkDestroyShaderModule(device, mergeFrag, nullptr);
    vkDestroyShaderModule(device, compressFrag, nullptr);
    vkDestroyShaderModule(device, registerFrag, nullptr);
    vkDestroyShaderModule(device, resolveFrag, nullptr);
    vkDestroyShaderModule(device, compositeFrag, nullptr);

    std::cout << "[UF] 8 pipelines created." << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createImage(VulkanContext& context, uint32_t w, uint32_t h,
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
        std::cerr << "[UF] Failed to create image" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[UF] Failed to allocate image memory" << std::endl;
        return false;
    }
    vkBindImageMemory(context.device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        std::cerr << "[UF] Failed to create image view" << std::endl;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createBuffer(VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage,
                              VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "[UF] Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[UF] Failed to allocate buffer memory" << std::endl;
        return false;
    }
    vkBindBufferMemory(context.device, buffer, memory, 0);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: FIND MEMORY TYPE
// ═══════════════════════════════════════════════════════════════════════════════

uint32_t UFRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                    VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    std::cerr << "[UF] No suitable memory type found!" << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESTROY SIZE-DEPENDENT RESOURCES
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::destroySizeDependent(VkDevice device) {
    if (m_headPointerView)   { vkDestroyImageView(device, m_headPointerView, nullptr); m_headPointerView = VK_NULL_HANDLE; }
    if (m_headPointerImage)  { vkDestroyImage(device, m_headPointerImage, nullptr); m_headPointerImage = VK_NULL_HANDLE; }
    if (m_headPointerMemory) { vkFreeMemory(device, m_headPointerMemory, nullptr); m_headPointerMemory = VK_NULL_HANDLE; }

    if (m_nodeBuffer)       { vkDestroyBuffer(device, m_nodeBuffer, nullptr); m_nodeBuffer = VK_NULL_HANDLE; }
    if (m_nodeBufferMemory) { vkFreeMemory(device, m_nodeBufferMemory, nullptr); m_nodeBufferMemory = VK_NULL_HANDLE; }
    m_nodeCapacity = 0;

    if (m_atomicCounterBuffer) { vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr); m_atomicCounterBuffer = VK_NULL_HANDLE; }
    if (m_atomicCounterMemory) { vkFreeMemory(device, m_atomicCounterMemory, nullptr); m_atomicCounterMemory = VK_NULL_HANDLE; }

    if (m_parentBuffer)       { vkDestroyBuffer(device, m_parentBuffer, nullptr); m_parentBuffer = VK_NULL_HANDLE; }
    if (m_parentBufferMemory) { vkFreeMemory(device, m_parentBufferMemory, nullptr); m_parentBufferMemory = VK_NULL_HANDLE; }

    // Root table
    if (m_rootTableBuffer)       { vkDestroyBuffer(device, m_rootTableBuffer, nullptr); m_rootTableBuffer = VK_NULL_HANDLE; }
    if (m_rootTableBufferMemory) { vkFreeMemory(device, m_rootTableBufferMemory, nullptr); m_rootTableBufferMemory = VK_NULL_HANDLE; }

    for (auto& layer : m_layerImages) {
        if (layer.framebuffer) { vkDestroyFramebuffer(device, layer.framebuffer, nullptr); layer.framebuffer = VK_NULL_HANDLE; }
        if (layer.view)        { vkDestroyImageView(device, layer.view, nullptr); layer.view = VK_NULL_HANDLE; }
        if (layer.image)       { vkDestroyImage(device, layer.image, nullptr); layer.image = VK_NULL_HANDLE; }
        if (layer.memory)      { vkFreeMemory(device, layer.memory, nullptr); layer.memory = VK_NULL_HANDLE; }
    }
    m_layerImages.clear();

    if (m_noAttachmentFramebuffer) { vkDestroyFramebuffer(device, m_noAttachmentFramebuffer, nullptr); m_noAttachmentFramebuffer = VK_NULL_HANDLE; }
    for (auto fb : m_compositeFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    m_compositeFramebuffers.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND RECORDING — MAIN ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCommands(VkCommandBuffer cmd,
                                VulkanContext& context,
                                SwapchainManager& swapchain,
                                VkBuffer vertexBuffer,
                                VkBuffer indexBuffer,
                                const std::vector<SceneObject>& objects,
                                const glm::mat4& viewProj,
                                uint32_t swapchainImageIndex) {

    // Read previous frame's timestamps (if available)
    if (enableTiming && m_timingReady) {
        uint64_t results[MAX_TIMESTAMPS];
        VkResult res = vkGetQueryPoolResults(context.device, m_timestampQueryPool,
            0, m_timestampCount, sizeof(results), results, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (res == VK_SUCCESS) {
            memcpy(m_timestamps, results, sizeof(uint64_t) * m_timestampCount);
            computeTimingResults();
        }
    }

    m_timestampCount = 0;

    // Reset queries for this frame
    if (enableTiming) {
        vkCmdResetQueryPool(cmd, m_timestampQueryPool, 0, MAX_TIMESTAMPS);
    }

    auto stamp = [&]() {
        if (enableTiming && m_timestampCount < MAX_TIMESTAMPS) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                m_timestampQueryPool, m_timestampCount++);
        }
    };

    // ─── One-time setup passes ───
    stamp(); // [0] start

    recordClearPhase(cmd);
    stamp(); // [1] after clear

    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [2] after collect

    recordSortPhase(cmd);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [3] after sort

    recordInitParentsPhase(cmd);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [4] after init parents

    recordMergePhase(cmd);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [5] after merge

    recordCompressPhase(cmd);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [6] after compress

    // ─── Peel loop ───
    for (uint32_t peel = 0; peel < NUM_LAYERS; peel++) {
        vkCmdFillBuffer(cmd, m_rootTableBuffer, 0,
                        MAX_GLOBAL_ROOTS * sizeof(uint32_t), 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, m_rootTableBuffer, MAX_GLOBAL_ROOTS * sizeof(uint32_t),
                        MAX_GLOBAL_ROOTS * sizeof(uint32_t), 0);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        recordRegisterRootsPhase(cmd);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        recordResolvePhase(cmd, peel);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        stamp(); // [7], [8], [9], [10] after each peel
    }

    // ─── Composite ───
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    recordCompositePhase(cmd, swapchainImageIndex);
    stamp(); // [11] after composite

    m_timingReady = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 1: CLEAR
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordClearPhase(VkCommandBuffer cmd) {
    // Transition head pointer image to TRANSFER_DST
    VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imgBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.image = m_headPointerImage;
    imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    // Clear head pointers to 0xFFFFFFFF (null)
    VkClearColorValue headClear{};
    headClear.uint32[0] = UF_NULL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_headPointerImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &headClear, 1, &range);

    // Reset atomic counter to 0
    vkCmdFillBuffer(cmd, m_atomicCounterBuffer, 0, sizeof(uint32_t), 0);

    // Transition head pointer image back to GENERAL
    imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkMemoryBarrier memBar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &memBar, 0, nullptr, 1, &imgBarrier);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2: COLLECT
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCollectPhase(VkCommandBuffer cmd,
                                    VkBuffer vertexBuffer, VkBuffer indexBuffer,
                                    const std::vector<SceneObject>& objects,
                                    const glm::mat4& viewProj) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipeline);

    // Flipped viewport for correct coordinate system
    VkViewport vp{0, (float)m_height, (float)m_width, -(float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_collectPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto& obj : objects) {
        UFScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = viewProj;
        pc.objectID = obj.objectID;
        pc.nearPlane = nearPlane;
        pc.farPlane = farPlane;
        vkCmdPushConstants(cmd, m_collectPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: SORT
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordSortPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_sortPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 4: INIT PARENTS
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordInitParentsPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_initParentsPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 5: MERGE (Union-Find)
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordMergePhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_mergePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 6: COMPRESS (Path Compression)
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCompressPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compressPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void UFRenderer::recordRegisterRootsPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_registerRootsPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 7: RESOLVE (one call per layer)
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex) {
    VkClearValue clear{};
    clear.color = { {0.0f, 0.0f, 0.0f, 0.0f} };

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_layerRenderPass;
    rp.framebuffer = m_layerImages[layerIndex].framebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resolvePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.depthThreshold = depthThreshold;
    pc.colorThreshold = colorThreshold;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;

    // Debug mode: layer 0 gets the debug code, other layers get normal (will output transparent)
    if (debugDisplayLayer >= 100 && layerIndex == 0) {
        pc.targetLayer = static_cast<uint32_t>(debugDisplayLayer);
    } else if (debugDisplayLayer >= 100) {
        pc.targetLayer = layerIndex;  // layers 1-3 will just output transparent in normal mode
    } else {
        pc.targetLayer = layerIndex;
    }

    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 8: COMPOSITE
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex) {
    VkClearValue clear{};
    clear.color = { {0.02f, 0.02f, 0.08f, 1.0f} };

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_compositeRenderPass;
    rp.framebuffer = m_compositeFramebuffers[swapchainImageIndex];
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_compositePipelineLayout, 0, 1, &m_compositeDescSet, 0, nullptr);

    UFCompositePushConstants pc{};
    pc.numLayers = NUM_LAYERS;
    pc.displayLayer = debugDisplayLayer;
    vkCmdPushConstants(cmd, m_compositePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: MEMORY BARRIER
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordMemoryBarrier(VkCommandBuffer cmd,
                                     VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                     VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void UFRenderer::computeTimingResults() {
    auto ms = [&](uint32_t start, uint32_t end) -> float {
        if (end >= m_timestampCount || start >= m_timestampCount) {
            return 0.0f;
        }
        uint64_t diff = m_timestamps[end] - m_timestamps[start];
        return float(diff) * m_timestampPeriod / 1000000.0f; // ticks → ns → ms
    };

    timingResults.clear     = ms(0, 1);
    timingResults.collect   = ms(1, 2);
    timingResults.sort      = ms(2, 3);
    timingResults.initParents = ms(3, 4);
    timingResults.merge     = ms(4, 5);
    timingResults.compress  = ms(5, 6);

    // Peels: timestamps 6→7, 7→8, 8→9, 9→10
    timingResults.peelTotal = 0;
    for (uint32_t i = 0; i < NUM_LAYERS; i++) {
        timingResults.peels[i] = ms(6 + i, 7 + i);
        timingResults.peelTotal += timingResults.peels[i];
    }

    timingResults.composite = ms(6 + NUM_LAYERS, 7 + NUM_LAYERS);
    timingResults.total     = ms(0, m_timestampCount - 1);
}