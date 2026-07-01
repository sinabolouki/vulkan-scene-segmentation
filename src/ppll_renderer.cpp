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
              << (m_nodeCapacity * sizeof(FragmentNode)) / (1024 * 1024) << " MB)" << std::endl;
    return true;
}

void PPLLRenderer::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_collectPipeline, nullptr);
    vkDestroyPipeline(device, m_sortPipeline, nullptr);
    vkDestroyPipeline(device, m_findMinPipeline, nullptr);
    vkDestroyPipeline(device, m_updateStatePipeline, nullptr);
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
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_R32_UINT;
        imgInfo.extent = {m_width, m_height, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(context.device, &imgInfo, nullptr, &m_headPointerImage) != VK_SUCCESS) return false;

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(context.device, m_headPointerImage, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(context.device, &allocInfo, nullptr, &m_headPointerMemory) != VK_SUCCESS) return false;
        vkBindImageMemory(context.device, m_headPointerImage, m_headPointerMemory, 0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_headPointerImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R32_UINT;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(context.device, &viewInfo, nullptr, &m_headPointerView) != VK_SUCCESS) return false;
    }

    // Node buffer
    m_nodeCapacity = m_width * m_height * AVG_FRAGS_PER_PIXEL;
    if (!createBuffer(context, m_nodeCapacity * sizeof(FragmentNode),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_nodeBuffer, m_nodeBufferMemory)) return false;

    // Atomic counter
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_atomicCounterBuffer, m_atomicCounterMemory)) return false;

    // State buffer: { uint globalMinUint, uint consumedDepthBits }
    if (!createBuffer(context, sizeof(uint32_t) * 2,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_stateBuffer, m_stateBufferMemory)) return false;

    return true;
}

bool PPLLRenderer::createLayerImages(VulkanContext& context) {
    m_layerImages.resize(NUM_PEEL_LAYERS);
    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
        if (!createImage(context, m_width, m_height,
                         VK_FORMAT_R8G8B8A8_UNORM,
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
    // No-attachment render pass (collect, sort, find-min, update-state)
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_noAttachmentRenderPass) != VK_SUCCESS) return false;
    }

    // Layer render pass (resolve → RGBA8 texture)
    {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAtt;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_layerRenderPass) != VK_SUCCESS) return false;
    }

    // Composite render pass (→ swapchain)
    {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = swapchainFormat;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAtt;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_compositeRenderPass) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAMEBUFFERS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    // No-attachment framebuffer
    {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_noAttachmentRenderPass;
        fbInfo.attachmentCount = 0;
        fbInfo.width = m_width;
        fbInfo.height = m_height;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_noAttachmentFramebuffer) != VK_SUCCESS) return false;
    }

    // Layer framebuffers
    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_layerRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &m_layerImages[i].view;
        fbInfo.width = m_width;
        fbInfo.height = m_height;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_layerImages[i].framebuffer) != VK_SUCCESS) return false;
    }

    // Composite framebuffers
    m_compositeFramebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        VkImageView view = swapchain.getImageView(i);
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_compositeRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &view;
        fbInfo.width = swapchain.getExtent().width;
        fbInfo.height = swapchain.getExtent().height;
        fbInfo.layers = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_compositeFramebuffers[i]) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createSampler(VkDevice device) {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_NEAREST;
    info.minFilter = VK_FILTER_NEAREST;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.maxLod = 1.0f;
    return vkCreateSampler(device, &info, nullptr, &m_sampler) == VK_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR SET LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createDescriptorSetLayouts(VkDevice device) {
    // PPLL set: 0=head image, 1=nodes, 2=counter, 3=state
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 4;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_ppllDescLayout) != VK_SUCCESS) return false;
    }

    // Composite set: binding 0..N-1 = layer color textures
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(NUM_PEEL_LAYERS);
        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = NUM_PEEL_LAYERS;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_compositeDescLayout) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR POOL & SETS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 3> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}; // nodes + counter + state
    sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NUM_PEEL_LAYERS};

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = static_cast<uint32_t>(sizes.size());
    info.pPoolSizes = sizes.data();
    info.maxSets = 2; // ppll set + composite set
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool PPLLRenderer::createDescriptorSets(VkDevice device) {
    // PPLL descriptor set
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_ppllDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_ppllDescSet) != VK_SUCCESS) return false;

        VkDescriptorImageInfo headInfo{};
        headInfo.imageView = m_headPointerView;
        headInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo nodeInfo{};
        nodeInfo.buffer = m_nodeBuffer;
        nodeInfo.offset = 0;
        nodeInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo counterInfo{};
        counterInfo.buffer = m_atomicCounterBuffer;
        counterInfo.offset = 0;
        counterInfo.range = sizeof(uint32_t);

        VkDescriptorBufferInfo stateInfo{};
        stateInfo.buffer = m_stateBuffer;
        stateInfo.offset = 0;
        stateInfo.range = sizeof(uint32_t) * 2;

        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ppllDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &headInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ppllDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeInfo, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ppllDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterInfo, nullptr};
        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ppllDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &stateInfo, nullptr};

        vkUpdateDescriptorSets(device, 4, writes.data(), 0, nullptr);
    }

    // Composite descriptor set
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_compositeDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_compositeDescSet) != VK_SUCCESS) return false;

        std::vector<VkDescriptorImageInfo> imgInfos(NUM_PEEL_LAYERS);
        std::vector<VkWriteDescriptorSet> writes(NUM_PEEL_LAYERS);
        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++) {
            imgInfos[i].sampler = m_sampler;
            imgInfos[i].imageView = m_layerImages[i].view;
            imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_compositeDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, NUM_PEEL_LAYERS, writes.data(), 0, nullptr);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createPipelineLayouts(VkDevice device) {
    // Collect: push = ScenePushConstants, set = ppll
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(ScenePushConstants);

        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_collectPipelineLayout) != VK_SUCCESS) return false;
    }

    // Fullscreen (sort, find-min, resolve, update-state): push = float depthThreshold, set = ppll
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(float);

        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_fullscreenPipelineLayout) != VK_SUCCESS) return false;
    }

    // Composite: push = uint numLayers, set = composite
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = 2 * sizeof(uint32_t); // numLayers + displayLayer

        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_compositeDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_compositePipelineLayout) != VK_SUCCESS) return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINES
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createPipelines(VkDevice device) {
    std::string sd = "shaders";

    VkShaderModule collectVert   = ShaderUtils::createShaderModule(device, sd + "/ppll_collect.vert.spv");
    VkShaderModule collectFrag   = ShaderUtils::createShaderModule(device, sd + "/ppll_collect.frag.spv");
    VkShaderModule fullscreenVert = ShaderUtils::createShaderModule(device, sd + "/ppll_resolve.vert.spv");
    VkShaderModule sortFrag      = ShaderUtils::createShaderModule(device, sd + "/ppll_sort.frag.spv");
    VkShaderModule findMinFrag   = ShaderUtils::createShaderModule(device, sd + "/ppll_find_min.frag.spv");
    VkShaderModule updateFrag    = ShaderUtils::createShaderModule(device, sd + "/ppll_update_state.frag.spv");
    VkShaderModule resolveFrag   = ShaderUtils::createShaderModule(device, sd + "/ppll_resolve.frag.spv");
    VkShaderModule compositeFrag = ShaderUtils::createShaderModule(device, sd + "/ppll_composite.frag.spv");

    if (!collectVert || !collectFrag || !fullscreenVert || !sortFrag ||
        !findMinFrag || !updateFrag || !resolveFrag || !compositeFrag) {
        std::cerr << "[PPLL] Failed to load one or more shaders!" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription vtxBinding = Vertex::getBindingDescription();
    auto vtxAttributes = Vertex::getAttributeDescriptions();
    PipelineBuilder builder;

    // Collect
    m_collectPipeline = builder
        .setVertexShader(collectVert).setFragmentShader(collectFrag)
        .setVertexInput({vtxBinding}, vtxAttributes)
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_collectPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_collectPipeline) return false;

    // Sort
    builder.clear();
    m_sortPipeline = builder
        .setVertexShader(fullscreenVert).setFragmentShader(sortFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_sortPipeline) return false;

    // Find-min
    builder.clear();
    m_findMinPipeline = builder
        .setVertexShader(fullscreenVert).setFragmentShader(findMinFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_findMinPipeline) return false;

    // Update-state
    builder.clear();
    m_updateStatePipeline = builder
        .setVertexShader(fullscreenVert).setFragmentShader(updateFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0).build(device);
    if (!m_updateStatePipeline) return false;

    // Resolve
    builder.clear();
    m_resolvePipeline = builder
        .setVertexShader(fullscreenVert).setFragmentShader(resolveFrag)
        .setNoVertexInput().setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest().disableBlending()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_layerRenderPass, 0).build(device);
    if (!m_resolvePipeline) return false;

    // Composite
    builder.clear();
    m_compositePipeline = builder
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
    vkDestroyShaderModule(device, updateFrag, nullptr);
    vkDestroyShaderModule(device, resolveFrag, nullptr);
    vkDestroyShaderModule(device, compositeFrag, nullptr);

    std::cout << "[PPLL] All 6 pipelines created." << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND RECORDING — MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordCommands(VkCommandBuffer cmd,
                                   VulkanContext& context,
                                   SwapchainManager& swapchain,
                                   VkBuffer vertexBuffer,
                                   VkBuffer indexBuffer,
                                   const std::vector<SceneObject>& objects,
                                   const glm::mat4& viewProj,
                                   uint32_t swapchainImageIndex) {
    // 1. Clear
    recordClearPhase(cmd);

    // 2. Collect
    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj);

    // 3. Barrier: collect writes → sort reads/writes
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // 4. Sort
    recordSortPhase(cmd);

    // 5. Barrier: sort writes → find-min reads
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // 6. Iterative peel
    for (uint32_t layer = 0; layer < NUM_PEEL_LAYERS; layer++) {
        // a. Find global min
        recordFindMinPhase(cmd);

        // b. Barrier: find-min writes state → resolve reads state
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        // c. Resolve layer
        recordResolvePhase(cmd, layer);

        // d. Barrier: resolve reads → update-state writes
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        // e. Update state
        recordUpdateStatePhase(cmd);

        // f. Barrier: update-state writes → next find-min reads
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    // 7. Composite
    recordCompositePhase(cmd, swapchainImageIndex);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordClearPhase(VkCommandBuffer cmd) {
    // Transition head image to TRANSFER_DST
    VkImageMemoryBarrier imgBarrier{};
    imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.image = m_headPointerImage;
    imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    // Clear head image
    VkClearColorValue clearVal{};
    clearVal.uint32[0] = PPLL_NULL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_headPointerImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &range);

    // Reset counter to 0
    vkCmdFillBuffer(cmd, m_atomicCounterBuffer, 0, sizeof(uint32_t), 0);

    // Reset state buffer: globalMinUint = 0xFFFFFFFF, consumedDepthBits = 0 (float 0.0)
    uint32_t stateInit[2] = {PPLL_NULL, 0};
    vkCmdUpdateBuffer(cmd, m_stateBuffer, 0, sizeof(stateInit), stateInit);

    // Transition head image to GENERAL
    imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkMemoryBarrier memBar{};
    memBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &memBar, 0, nullptr, 1, &imgBarrier);
}

void PPLLRenderer::recordCollectPhase(VkCommandBuffer cmd, VkBuffer vertexBuffer, VkBuffer indexBuffer,
                                       const std::vector<SceneObject>& objects, const glm::mat4& viewProj) {
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_noAttachmentRenderPass;
    rpInfo.framebuffer = m_noAttachmentFramebuffer;
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
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

void PPLLRenderer::recordSortPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_noAttachmentRenderPass;
    rpInfo.framebuffer = m_noAttachmentFramebuffer;
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_sortPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &depthThreshold);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void PPLLRenderer::recordFindMinPhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_noAttachmentRenderPass;
    rpInfo.framebuffer = m_noAttachmentFramebuffer;
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_findMinPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &depthThreshold);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void PPLLRenderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex) {
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_layerRenderPass;
    rpInfo.framebuffer = m_layerImages[layerIndex].framebuffer;
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resolvePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &depthThreshold);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void PPLLRenderer::recordUpdateStatePhase(VkCommandBuffer cmd) {
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_noAttachmentRenderPass;
    rpInfo.framebuffer = m_noAttachmentFramebuffer;
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_updateStatePipeline);

    // Use 1x1 scissor — only one pixel needs to execute
    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0, 0}, {1, 1}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &depthThreshold);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void PPLLRenderer::recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex) {
    VkClearValue clear{};
    clear.color = {{0.02f, 0.02f, 0.08f, 1.0f}};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_compositeRenderPass;
    rpInfo.framebuffer = m_compositeFramebuffers[swapchainImageIndex];
    rpInfo.renderArea = {{0, 0}, {m_width, m_height}};
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = {{0,0}, {m_width, m_height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipelineLayout, 0, 1, &m_compositeDescSet, 0, nullptr);
    struct CompositePushConstants {
        uint32_t numLayers;
        int32_t  displayLayer; // -1 = composite all, 0..N-1 = single layer
    } compositePC;
    compositePC.numLayers = NUM_PEEL_LAYERS;
    compositePC.displayLayer = debugDisplayLayer;
    vkCmdPushConstants(cmd, m_compositePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(compositePC), &compositePC);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY BARRIER HELPER
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::recordMemoryBarrier(VkCommandBuffer cmd,
                                        VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                        VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd,
        srcStage, dstStage,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE CREATION HELPER
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createImage(VulkanContext& context, uint32_t w, uint32_t h,
                                VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
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
        std::cerr << "[PPLL] Failed to create image" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to allocate image memory" << std::endl;
        return false;
    }

    vkBindImageMemory(context.device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
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
// BUFFER CREATION HELPER
// ═══════════════════════════════════════════════════════════════════════════════

bool PPLLRenderer::createBuffer(VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
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
// FIND MEMORY TYPE
// ═══════════════════════════════════════════════════════════════════════════════

uint32_t PPLLRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cerr << "[PPLL] Failed to find suitable memory type!" << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESTROY SIZE-DEPENDENT RESOURCES
// ═══════════════════════════════════════════════════════════════════════════════

void PPLLRenderer::destroySizeDependent(VkDevice device) {
    // Head pointer image
    if (m_headPointerView) { vkDestroyImageView(device, m_headPointerView, nullptr); m_headPointerView = VK_NULL_HANDLE; }
    if (m_headPointerImage) { vkDestroyImage(device, m_headPointerImage, nullptr); m_headPointerImage = VK_NULL_HANDLE; }
    if (m_headPointerMemory) { vkFreeMemory(device, m_headPointerMemory, nullptr); m_headPointerMemory = VK_NULL_HANDLE; }

    // Node buffer
    if (m_nodeBuffer) { vkDestroyBuffer(device, m_nodeBuffer, nullptr); m_nodeBuffer = VK_NULL_HANDLE; }
    if (m_nodeBufferMemory) { vkFreeMemory(device, m_nodeBufferMemory, nullptr); m_nodeBufferMemory = VK_NULL_HANDLE; }
    m_nodeCapacity = 0;

    // Atomic counter
    if (m_atomicCounterBuffer) { vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr); m_atomicCounterBuffer = VK_NULL_HANDLE; }
    if (m_atomicCounterMemory) { vkFreeMemory(device, m_atomicCounterMemory, nullptr); m_atomicCounterMemory = VK_NULL_HANDLE; }

    // State buffer
    if (m_stateBuffer) { vkDestroyBuffer(device, m_stateBuffer, nullptr); m_stateBuffer = VK_NULL_HANDLE; }
    if (m_stateBufferMemory) { vkFreeMemory(device, m_stateBufferMemory, nullptr); m_stateBufferMemory = VK_NULL_HANDLE; }

    // Layer images
    for (auto& layer : m_layerImages) {
        if (layer.framebuffer) { vkDestroyFramebuffer(device, layer.framebuffer, nullptr); layer.framebuffer = VK_NULL_HANDLE; }
        if (layer.view) { vkDestroyImageView(device, layer.view, nullptr); layer.view = VK_NULL_HANDLE; }
        if (layer.image) { vkDestroyImage(device, layer.image, nullptr); layer.image = VK_NULL_HANDLE; }
        if (layer.memory) { vkFreeMemory(device, layer.memory, nullptr); layer.memory = VK_NULL_HANDLE; }
    }
    m_layerImages.clear();

    // No-attachment framebuffer
    if (m_noAttachmentFramebuffer) { vkDestroyFramebuffer(device, m_noAttachmentFramebuffer, nullptr); m_noAttachmentFramebuffer = VK_NULL_HANDLE; }

    // Composite framebuffers
    for (auto fb : m_compositeFramebuffers) {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    m_compositeFramebuffers.clear();
}