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
#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT / DESTROY / RECREATE
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::init(VulkanContext &context, SwapchainManager &swapchain)
{
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    if (!createDataStructures(context))
        return false;
    if (!createLayerImages(context))
        return false;
    if (!createRenderPasses(context.device, swapchain.getImageFormat()))
        return false;
    if (!createFramebuffers(context.device, swapchain))
        return false;
    if (!createSampler(context.device))
        return false;
    if (!createDescriptorSetLayouts(context.device))
        return false;
    if (!createDescriptorPool(context.device))
        return false;
    if (!createDescriptorSets(context.device))
        return false;
    if (!createPipelineLayouts(context.device))
        return false;
    if (!createPipelines(context.device))
        return false;

    if (!createProfilingResources(context))
        return false;

    std::cout << "[UF] Initialized: " << m_width << "x" << m_height
              << ", " << NUM_LAYERS << " layers, "
              << m_nodeCapacity << " nodes ("
              << (m_nodeCapacity * sizeof(UFFragmentNode)) / (1024 * 1024) << " MB node buf, "
              << (m_nodeCapacity * sizeof(uint32_t)) / (1024 * 1024) << " MB parent buf)"
              << std::endl;
    return true;
}

void UFRenderer::destroy(VkDevice device)
{
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_collectPipeline, nullptr);
    vkDestroyPipeline(device, m_sortPipeline, nullptr);
    vkDestroyPipeline(device, m_initParentsPipeline, nullptr);
    vkDestroyPipeline(device, m_mergePipeline, nullptr);
    vkDestroyPipeline(device, m_compressPipeline, nullptr);
    vkDestroyPipeline(device, m_registerRootsPipeline, nullptr);
    vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    vkDestroyPipeline(device, m_compositePipeline, nullptr);
    vkDestroyPipeline(device, m_debugRegionsPipeline, nullptr);
    vkDestroyPipeline(device, m_findMinPipeline, nullptr);


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

    destroyProfilingResources(device);

    destroySizeDependent(device);
}

void UFRenderer::recreate(VulkanContext &context, SwapchainManager &swapchain)
{
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

bool UFRenderer::createDataStructures(VulkanContext &context)
{
    // Head pointer image (R32_UINT)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R32_UINT;
        info.extent = {m_width, m_height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(context.device, &info, nullptr, &m_headPointerImage) != VK_SUCCESS)
            return false;

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(context.device, m_headPointerImage, &memReq);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = memReq.size;
        alloc.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(context.device, &alloc, nullptr, &m_headPointerMemory) != VK_SUCCESS)
            return false;
        vkBindImageMemory(context.device, m_headPointerImage, m_headPointerMemory, 0);

        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = m_headPointerImage;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R32_UINT;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(context.device, &view, nullptr, &m_headPointerView) != VK_SUCCESS)
            return false;
    }

    // Node buffer (SSBO)
    m_nodeCapacity = m_width * m_height * AVG_FRAGS_PER_PIXEL;
    if (!createBuffer(context, m_nodeCapacity * sizeof(UFFragmentNode),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      m_nodeBuffer, m_nodeBufferMemory))
        return false;

    // Atomic counter (4 bytes)
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      m_atomicCounterBuffer, m_atomicCounterMemory))
        return false;

    // Parent buffer (one uint per node)
    if (!createBuffer(context, m_nodeCapacity * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      m_parentBuffer, m_parentBufferMemory))
        return false;

    // Root table (8 entries × 2 uints = 64 bytes)
    if (!createBuffer(context, m_nodeCapacity * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      m_rootMaxPosBuffer, m_rootMaxPosMemory))
        return false;

    // Global min buffer (single uint — holds minimum maxPosition across all roots)
    if (!createBuffer(context, sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      m_globalMinBuffer, m_globalMinMemory))
        return false;

    // Debug info buffer (GPU-side, written by shaders)
    {
        VkDeviceSize debugSize = sizeof(uint32_t) * 4;  // minMaxPos, numRoots, pad, pad

        // GPU buffer (written via atomics in shaders)
        if (!createBuffer(context, debugSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          m_debugInfoBuffer, m_debugInfoMemory))
            return false;

    }

    return true;
}

bool UFRenderer::createLayerImages(VulkanContext &context)
{
    m_layerImages.resize(NUM_LAYERS);
    for (uint32_t i = 0; i < NUM_LAYERS; i++)
    {
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

bool UFRenderer::createRenderPasses(VkDevice device, VkFormat swapchainFormat)
{
    // No-attachment render pass (for collect, sort, init, merge, compress)
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_noAttachmentRenderPass) != VK_SUCCESS)
            return false;
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
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 1;
        rp.pAttachments = &att;
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        rp.dependencyCount = 1;
        rp.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_layerRenderPass) != VK_SUCCESS)
            return false;
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
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 1;
        rp.pAttachments = &att;
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        rp.dependencyCount = 1;
        rp.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_compositeRenderPass) != VK_SUCCESS)
            return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAMEBUFFERS
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createFramebuffers(VkDevice device, SwapchainManager &swapchain)
{
    // No-attachment framebuffer
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_noAttachmentRenderPass;
        fb.attachmentCount = 0;
        fb.width = m_width;
        fb.height = m_height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_noAttachmentFramebuffer) != VK_SUCCESS)
            return false;
    }

    // Layer framebuffers (one per layer image)
    for (uint32_t i = 0; i < NUM_LAYERS; i++)
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_layerRenderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = &m_layerImages[i].view;
        fb.width = m_width;
        fb.height = m_height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_layerImages[i].framebuffer) != VK_SUCCESS)
            return false;
    }

    // Composite framebuffers (one per swapchain image)
    m_compositeFramebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++)
    {
        VkImageView view = swapchain.getImageView(i);
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_compositeRenderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = &view;
        fb.width = swapchain.getExtent().width;
        fb.height = swapchain.getExtent().height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_compositeFramebuffers[i]) != VK_SUCCESS)
            return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLER
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createSampler(VkDevice device)
{
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

bool UFRenderer::createDescriptorSetLayouts(VkDevice device)
{
    // PPLL set: 6 bindings
    {
        std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = (uint32_t)bindings.size();
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_ppllDescLayout) != VK_SUCCESS)
            return false;
    }

    // Composite set: NUM_LAYERS combined image samplers
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(NUM_LAYERS);
        for (uint32_t i = 0; i < NUM_LAYERS; i++)
        {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        }

        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = NUM_LAYERS;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &m_compositeDescLayout) != VK_SUCCESS)
            return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESCRIPTOR POOL & SETS
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createDescriptorPool(VkDevice device)
{
    std::array<VkDescriptorPoolSize, 3> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}; // nodes + counter + parent + rootMaxPos + globalMin + debug
    sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NUM_LAYERS};

    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount = (uint32_t)sizes.size();
    info.pPoolSizes = sizes.data();
    info.maxSets = 2;
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool UFRenderer::createDescriptorSets(VkDevice device)
{
    // PPLL descriptor set
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &m_ppllDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_ppllDescSet) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo headInfo{};
        headInfo.imageView = m_headPointerView;
        headInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo nodeInfo{m_nodeBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo counterInfo{m_atomicCounterBuffer, 0, sizeof(uint32_t)};
        VkDescriptorBufferInfo parentInfo{m_parentBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo rootMaxPosInfo{m_rootMaxPosBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo globalMinInfo{m_globalMinBuffer, 0, sizeof(uint32_t)};
        VkDescriptorBufferInfo debugInfoBufInfo{m_debugInfoBuffer, 0, VK_WHOLE_SIZE};

        std::array<VkWriteDescriptorSet, 7> writes{};

        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = m_ppllDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &headInfo;

        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = m_ppllDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &nodeInfo;

        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[2].dstSet = m_ppllDescSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &counterInfo;

        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[3].dstSet = m_ppllDescSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &parentInfo;

         writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[4].dstSet = m_ppllDescSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &rootMaxPosInfo;

        writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[5].dstSet = m_ppllDescSet;
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &globalMinInfo;

        writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[6].dstSet = m_ppllDescSet;
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &debugInfoBufInfo;

        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Composite descriptor set (unchanged)
    {
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = m_descriptorPool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &m_compositeDescLayout;
        if (vkAllocateDescriptorSets(device, &alloc, &m_compositeDescSet) != VK_SUCCESS)
            return false;

        std::vector<VkDescriptorImageInfo> imgInfos(NUM_LAYERS);
        std::vector<VkWriteDescriptorSet> writes(NUM_LAYERS);
        for (uint32_t i = 0; i < NUM_LAYERS; i++)
        {
            imgInfos[i].sampler = m_sampler;
            imgInfos[i].imageView = m_layerImages[i].view;
            imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = m_compositeDescSet;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, NUM_LAYERS, writes.data(), 0, nullptr);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createPipelineLayouts(VkDevice device)
{

    // Collect pipeline layout: set 0 = PPLL, set 1 = Textures (if available)
    {
        std::vector<VkDescriptorSetLayout> setLayouts;
        setLayouts.push_back(m_ppllDescLayout); // set 0

        if (m_textureManager)
        {
            setLayouts.push_back(m_textureManager->getDescriptorSetLayout()); // set 1
        }

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFScenePushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = (uint32_t)setLayouts.size();
        info.pSetLayouts = setLayouts.data();
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_collectPipelineLayout) != VK_SUCCESS)
            return false;
    }

    // Fullscreen pipeline layout: push = UFFullscreenPushConstants
    // Used by sort, init_parents, merge, compress, resolve
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFFullscreenPushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_fullscreenPipelineLayout) != VK_SUCCESS)
            return false;
    }

    // Composite pipeline layout: push = UFCompositePushConstants
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFCompositePushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts = &m_compositeDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_compositePipelineLayout) != VK_SUCCESS)
            return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINES
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createPipelines(VkDevice device)
{
    std::string sd = "shaders/union_find";

    VkShaderModule collectVert = ShaderUtils::createShaderModule(device, sd + "/uf_collect.vert.spv");
    VkShaderModule collectFrag = ShaderUtils::createShaderModule(device, sd + "/uf_collect.frag.spv");
    VkShaderModule fullscreenVert = ShaderUtils::createShaderModule(device, sd + "/uf_fullscreen.vert.spv");
    VkShaderModule sortFrag = ShaderUtils::createShaderModule(device, sd + "/uf_sort.frag.spv");
    VkShaderModule initFrag = ShaderUtils::createShaderModule(device, sd + "/uf_init_parents.frag.spv");
    VkShaderModule mergeFrag = ShaderUtils::createShaderModule(device, sd + "/uf_merge.frag.spv");
    VkShaderModule compressFrag = ShaderUtils::createShaderModule(device, sd + "/uf_compress.frag.spv");
    VkShaderModule registerFrag = ShaderUtils::createShaderModule(device, sd + "/uf_register_roots.frag.spv");
    VkShaderModule resolveFrag = ShaderUtils::createShaderModule(device, sd + "/uf_resolve.frag.spv");
    VkShaderModule compositeFrag = ShaderUtils::createShaderModule(device, sd + "/uf_composite.frag.spv");
    VkShaderModule findMinFrag = ShaderUtils::createShaderModule(device, sd + "/uf_find_min.frag.spv");
    VkShaderModule debugRegionsFrag = ShaderUtils::createShaderModule(device, sd + "/uf_debug_regions.frag.spv");


    if (!collectVert || !collectFrag || !fullscreenVert || !sortFrag ||
        !initFrag || !mergeFrag || !compressFrag || !registerFrag || !resolveFrag || !compositeFrag || !findMinFrag)
    {
        std::cerr << "[UF] Failed to load one or more shaders!" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription vtxBind = Vertex::getBindingDescription();
    auto vtxAttr = Vertex::getAttributeDescriptions();
    PipelineBuilder b;

    // ─── Collect ───
    m_collectPipeline = b
                            .setVertexShader(collectVert)
                            .setFragmentShader(collectFrag)
                            .setVertexInput({vtxBind}, vtxAttr)
                            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                            .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                            .disableDepthTest()
                            .setNoColorAttachments()
                            .setPipelineLayout(m_collectPipelineLayout)
                            .setRenderPass(m_noAttachmentRenderPass, 0)
                            .build(device);
    if (!m_collectPipeline)
        return false;

    // ─── Sort ───
    b.clear();
    m_sortPipeline = b
                         .setVertexShader(fullscreenVert)
                         .setFragmentShader(sortFrag)
                         .setNoVertexInput()
                         .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                         .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                         .disableDepthTest()
                         .setNoColorAttachments()
                         .setPipelineLayout(m_fullscreenPipelineLayout)
                         .setRenderPass(m_noAttachmentRenderPass, 0)
                         .build(device);
    if (!m_sortPipeline)
        return false;

    // ─── Init Parents ───
    b.clear();
    m_initParentsPipeline = b
                                .setVertexShader(fullscreenVert)
                                .setFragmentShader(initFrag)
                                .setNoVertexInput()
                                .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                                .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                .disableDepthTest()
                                .setNoColorAttachments()
                                .setPipelineLayout(m_fullscreenPipelineLayout)
                                .setRenderPass(m_noAttachmentRenderPass, 0)
                                .build(device);
    if (!m_initParentsPipeline)
        return false;

    // ─── Merge ───
    b.clear();
    m_mergePipeline = b
                          .setVertexShader(fullscreenVert)
                          .setFragmentShader(mergeFrag)
                          .setNoVertexInput()
                          .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                          .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                          .disableDepthTest()
                          .setNoColorAttachments()
                          .setPipelineLayout(m_fullscreenPipelineLayout)
                          .setRenderPass(m_noAttachmentRenderPass, 0)
                          .build(device);
    if (!m_mergePipeline)
        return false;

    // ─── Compress ───
    b.clear();
    m_compressPipeline = b
                             .setVertexShader(fullscreenVert)
                             .setFragmentShader(compressFrag)
                             .setNoVertexInput()
                             .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                             .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                             .disableDepthTest()
                             .setNoColorAttachments()
                             .setPipelineLayout(m_fullscreenPipelineLayout)
                             .setRenderPass(m_noAttachmentRenderPass, 0)
                             .build(device);
    if (!m_compressPipeline)
        return false;

    // ─── Register Roots ───
    b.clear();
    m_registerRootsPipeline = b
                                  .setVertexShader(fullscreenVert)
                                  .setFragmentShader(registerFrag)
                                  .setNoVertexInput()
                                  .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                                  .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                  .disableDepthTest()
                                  .setNoColorAttachments()
                                  .setPipelineLayout(m_fullscreenPipelineLayout)
                                  .setRenderPass(m_noAttachmentRenderPass, 0)
                                  .build(device);
    if (!m_registerRootsPipeline)
        return false;

    // ─── Resolve ───
    b.clear();
    m_resolvePipeline = b
                            .setVertexShader(fullscreenVert)
                            .setFragmentShader(resolveFrag)
                            .setNoVertexInput()
                            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                            .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                            .disableDepthTest()
                            .disableBlending()
                            .setPipelineLayout(m_fullscreenPipelineLayout)
                            .setRenderPass(m_layerRenderPass, 0)
                            .build(device);
    if (!m_resolvePipeline)
        return false;

    // ─── Find Min ───
    b.clear();
    m_findMinPipeline = b
        .setVertexShader(fullscreenVert)
        .setFragmentShader(findMinFrag)
        .setNoVertexInput()
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest()
        .setNoColorAttachments()
        .setPipelineLayout(m_fullscreenPipelineLayout)
        .setRenderPass(m_noAttachmentRenderPass, 0)
        .build(device);
    if (!m_findMinPipeline) return false;

    // ─── Composite ───
    b.clear();
    m_compositePipeline = b
                              .setVertexShader(fullscreenVert)
                              .setFragmentShader(compositeFrag)
                              .setNoVertexInput()
                              .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                              .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                              .disableDepthTest()
                              .disableBlending()
                              .setPipelineLayout(m_compositePipelineLayout)
                              .setRenderPass(m_compositeRenderPass, 0)
                              .build(device);
    if (!m_compositePipeline)
        return false;

    // ─── Debug Regions ───
    b.clear();
    m_debugRegionsPipeline = b
                                .setVertexShader(fullscreenVert)
                                .setFragmentShader(debugRegionsFrag)
                                .setNoVertexInput()
                                .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                                .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                .disableDepthTest()
                                .disableBlending()
                                .setPipelineLayout(m_fullscreenPipelineLayout)
                                .setRenderPass(m_compositeRenderPass, 0)
                                .build(device);
    if (!m_debugRegionsPipeline)
    return false;

    // Cleanup shader modules
    vkDestroyShaderModule(device, collectVert, nullptr);
    vkDestroyShaderModule(device, collectFrag, nullptr);
    vkDestroyShaderModule(device, fullscreenVert, nullptr);
    vkDestroyShaderModule(device, sortFrag, nullptr);
    vkDestroyShaderModule(device, initFrag, nullptr);
    vkDestroyShaderModule(device, mergeFrag, nullptr);
    vkDestroyShaderModule(device, compressFrag, nullptr);
    vkDestroyShaderModule(device, registerFrag, nullptr);
    vkDestroyShaderModule(device, findMinFrag, nullptr);
    vkDestroyShaderModule(device, debugRegionsFrag, nullptr);
    vkDestroyShaderModule(device, resolveFrag, nullptr);
    vkDestroyShaderModule(device, compositeFrag, nullptr);

    std::cout << "[UF] 8 pipelines created." << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createImage(VulkanContext &context, uint32_t w, uint32_t h,
                             VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                             VkImage &image, VkDeviceMemory &memory, VkImageView &view)
{
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

    if (vkCreateImage(context.device, &imgInfo, nullptr, &image) != VK_SUCCESS)
    {
        std::cerr << "[UF] Failed to create image" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
        std::cerr << "[UF] Failed to allocate image memory" << std::endl;
        return false;
    }
    vkBindImageMemory(context.device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
    {
        std::cerr << "[UF] Failed to create image view" << std::endl;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: CREATE BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

bool UFRenderer::createBuffer(VulkanContext &context, VkDeviceSize size, VkBufferUsageFlags usage,
                              VkBuffer &buffer, VkDeviceMemory &memory)
{
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        std::cerr << "[UF] Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
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
                                    VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    std::cerr << "[UF] No suitable memory type found!" << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESTROY SIZE-DEPENDENT RESOURCES
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::destroySizeDependent(VkDevice device)
{
    if (!m_evaluationPath.empty())
        m_evaluationStatus = "Evaluation export canceled during resource recreation.";
    destroyEvaluationReadback(device);
    m_evaluationPath.clear();

    if (m_headPointerView)
    {
        vkDestroyImageView(device, m_headPointerView, nullptr);
        m_headPointerView = VK_NULL_HANDLE;
    }
    if (m_headPointerImage)
    {
        vkDestroyImage(device, m_headPointerImage, nullptr);
        m_headPointerImage = VK_NULL_HANDLE;
    }
    if (m_headPointerMemory)
    {
        vkFreeMemory(device, m_headPointerMemory, nullptr);
        m_headPointerMemory = VK_NULL_HANDLE;
    }

    if (m_nodeBuffer)
    {
        vkDestroyBuffer(device, m_nodeBuffer, nullptr);
        m_nodeBuffer = VK_NULL_HANDLE;
    }
    if (m_nodeBufferMemory)
    {
        vkFreeMemory(device, m_nodeBufferMemory, nullptr);
        m_nodeBufferMemory = VK_NULL_HANDLE;
    }
    m_nodeCapacity = 0;

    if (m_atomicCounterBuffer)
    {
        vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr);
        m_atomicCounterBuffer = VK_NULL_HANDLE;
    }
    if (m_atomicCounterMemory)
    {
        vkFreeMemory(device, m_atomicCounterMemory, nullptr);
        m_atomicCounterMemory = VK_NULL_HANDLE;
    }

    if (m_parentBuffer)
    {
        vkDestroyBuffer(device, m_parentBuffer, nullptr);
        m_parentBuffer = VK_NULL_HANDLE;
    }
    if (m_parentBufferMemory)
    {
        vkFreeMemory(device, m_parentBufferMemory, nullptr);
        m_parentBufferMemory = VK_NULL_HANDLE;
    }

    // Root max position buffer
    if (m_rootMaxPosBuffer) {
        vkDestroyBuffer(device, m_rootMaxPosBuffer, nullptr);
        m_rootMaxPosBuffer = VK_NULL_HANDLE;
    }
    if (m_rootMaxPosMemory) {
        vkFreeMemory(device, m_rootMaxPosMemory, nullptr);
        m_rootMaxPosMemory = VK_NULL_HANDLE;
    }

    // Global min buffer
    if (m_globalMinBuffer) {
        vkDestroyBuffer(device, m_globalMinBuffer, nullptr);
        m_globalMinBuffer = VK_NULL_HANDLE;
    }
    if (m_globalMinMemory) {
        vkFreeMemory(device, m_globalMinMemory, nullptr);
        m_globalMinMemory = VK_NULL_HANDLE;
    }

    // Global min staging buffer
    if (m_globalMinStagingBuffer) {
        if (m_globalMinStagingMapped) {
            vkUnmapMemory(device, m_globalMinStagingMemory);
            m_globalMinStagingMapped = nullptr;
        }
        vkDestroyBuffer(device, m_globalMinStagingBuffer, nullptr);
        m_globalMinStagingBuffer = VK_NULL_HANDLE;
    }
    if (m_globalMinStagingMemory) {
        vkFreeMemory(device, m_globalMinStagingMemory, nullptr);
        m_globalMinStagingMemory = VK_NULL_HANDLE;
    }

    if (m_debugInfoBuffer) {
        vkDestroyBuffer(device, m_debugInfoBuffer, nullptr);
        m_debugInfoBuffer = VK_NULL_HANDLE;
    }
    if (m_debugInfoMemory) {
        vkFreeMemory(device, m_debugInfoMemory, nullptr);
        m_debugInfoMemory = VK_NULL_HANDLE;
    }
    for (auto &layer : m_layerImages)
    {
        if (layer.framebuffer)
        {
            vkDestroyFramebuffer(device, layer.framebuffer, nullptr);
            layer.framebuffer = VK_NULL_HANDLE;
        }
        if (layer.view)
        {
            vkDestroyImageView(device, layer.view, nullptr);
            layer.view = VK_NULL_HANDLE;
        }
        if (layer.image)
        {
            vkDestroyImage(device, layer.image, nullptr);
            layer.image = VK_NULL_HANDLE;
        }
        if (layer.memory)
        {
            vkFreeMemory(device, layer.memory, nullptr);
            layer.memory = VK_NULL_HANDLE;
        }
    }
    m_layerImages.clear();

    if (m_noAttachmentFramebuffer)
    {
        vkDestroyFramebuffer(device, m_noAttachmentFramebuffer, nullptr);
        m_noAttachmentFramebuffer = VK_NULL_HANDLE;
    }
    for (auto fb : m_compositeFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    m_compositeFramebuffers.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND RECORDING — MAIN ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCommands(VkCommandBuffer cmd,
                                VulkanContext &context,
                                SwapchainManager &swapchain,
                                VkBuffer vertexBuffer,
                                VkBuffer indexBuffer,
                                const std::vector<SceneObject> &objects,
                                const glm::mat4 &viewProj,
                                const glm::mat4 &view,
                                const glm::mat4 &projection,
                                uint32_t swapchainImageIndex,
                                uint32_t frameSlot, uint64_t applicationFrameId)
{
    m_projX = projection[0][0];
    m_projY = projection[1][1];

    const bool useDebugVis = (debugVisualizationMode >= 0);
    const bool captureThisFrame =
        !m_evaluationPath.empty() && prepareEvaluationReadback(context);
    const bool measureThisFrame =
        enableTiming && m_timestampQueryPool != VK_NULL_HANDLE &&
        !useDebugVis && !captureThisFrame;
    if (captureThisFrame) {
        timingResults = {};
        m_timingReady = false;
    }

    auto& profile = m_profileFrames.at(frameSlot);
    if (profile.pending)
        throw std::logic_error("Profiling slot reused before its fence was harvested");
    profile.frameId = applicationFrameId;
    profile.nodeCapacity = m_nodeCapacity;
    profile.timestampCount = 0;
    profile.layered = !useDebugVis;
    profile.grouping = !bypassUnionFind && mergeIterations > 0;
    profile.evaluationExport = captureThisFrame;
    profile.timed = measureThisFrame;
    const uint32_t queryBase = frameSlot * MAX_TIMESTAMPS;
    if (profile.timed)
        vkCmdResetQueryPool(cmd, m_timestampQueryPool, queryBase, MAX_TIMESTAMPS);

    // PPLL nodes, heads and layer images are shared by normal flight slots.
    // Order their reuse, including the preceding slot's diagnostic transfer
    // reads. This is a queue dependency, not a host wait or a frame-count change.
    // It is before the start timestamp so older-frame work is not timed here.
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    static_assert(7 + 3 * NUM_LAYERS <= MAX_TIMESTAMPS, "Timestamp capacity too small");

    auto stamp = [&]() {
        if (profile.timed) {
            if (profile.timestampCount >= MAX_TIMESTAMPS)
                throw std::logic_error("Timestamp capacity exceeded");
            // TOP brackets the start of this renderer; BOTTOM markers measure
            // completion of each dependent phase, including its barriers.
            const VkPipelineStageFlagBits stage = profile.timestampCount == 0
                ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            vkCmdWriteTimestamp(cmd, stage, m_timestampQueryPool,
                                queryBase + profile.timestampCount++);
        }
    };

    // ═════════════════════════════════════════════════════════════════════
    // PHASE 1: CLEAR
    // ═════════════════════════════════════════════════════════════════════
    stamp(); // [0] frame start

    recordClearPhase(cmd);
    stamp(); // [1] after clear

    // ═════════════════════════════════════════════════════════════════════
    // PHASE 2: COLLECT
    // ═════════════════════════════════════════════════════════════════════
    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj, view);
    recordMemoryBarrier(cmd,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);

    stamp(); // [2] after collect

    // ═════════════════════════════════════════════════════════════════════
    // PHASE 3: SORT
    // ═════════════════════════════════════════════════════════════════════
    recordSortPhase(cmd);
    recordMemoryBarrier(cmd,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [3] after sort

    // ═════════════════════════════════════════════════════════════════════
    // PHASE 4: INIT PARENTS
    // ═════════════════════════════════════════════════════════════════════
    recordInitParentsPhase(cmd);
    recordMemoryBarrier(cmd,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    stamp(); // [4] after init parents

    // ═════════════════════════════════════════════════════════════════════
    // PHASE 5 & 6: MERGE + COMPRESS (iterative, skipped if bypass)
    // ═════════════════════════════════════════════════════════════════════
    if (!bypassUnionFind)
    {
        for (uint32_t iter = 0; iter < mergeIterations; iter++)
        {
            recordMergePhase(cmd);
            recordMemoryBarrier(cmd,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            recordCompressPhase(cmd);
            recordMemoryBarrier(cmd,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }
    stamp(); // [5] end of ALL merge-compression iterations, including barriers

    if (captureThisFrame)
        recordEvaluationReadback(cmd, static_cast<uint32_t>(mergeIterations));

    // ═════════════════════════════════════════════════════════════════════
    // OUTPUT: Either debug visualization or peel+composite
    // ═════════════════════════════════════════════════════════════════════
    if (useDebugVis)
    {
        // Clear rootMaxPos buffer
        vkCmdFillBuffer(cmd, m_rootMaxPosBuffer, 0,
                        m_nodeCapacity * sizeof(uint32_t), 0);

        // Clear globalMin
        vkCmdFillBuffer(cmd, m_globalMinBuffer, 0, sizeof(uint32_t), 0xFFFFFFFF);

        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        // Register roots
        recordRegisterRootsPhase(cmd);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        // Find global min (needed for debug mode 6)
        recordFindMinPhase(cmd);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);


        // Render debug directly to swapchain
        recordDebugRegionsPhase(cmd, swapchainImageIndex);
        stamp(); // [6] end of debug output (no extraction on this path)
    }
    else
    {
        // ─── Normal path: peel loop + composite ──────────────────────────

        // Clear debug info buffer before peeling
        vkCmdFillBuffer(cmd, m_debugInfoBuffer, 0, sizeof(uint32_t), 0xFFFFFFFF);  // debugMinMaxPos
        vkCmdFillBuffer(cmd, m_debugInfoBuffer, sizeof(uint32_t), sizeof(uint32_t), 0);  // debugNumRoots
        vkCmdFillBuffer(cmd, m_debugInfoBuffer, 2 * sizeof(uint32_t), 2 * sizeof(uint32_t), 0);  // padding
        recordMemoryBarrier(cmd,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        // ─── Peel loop ───────────────────────────────────────────────────
        for (uint32_t peel = 0; peel < NUM_LAYERS; peel++)
        {
            // Clear rootMaxPosition buffer to 0
            vkCmdFillBuffer(cmd, m_rootMaxPosBuffer, 0,
                            m_nodeCapacity * sizeof(uint32_t), 0);

            // Clear globalMin to 0xFFFFFFFF (for atomicMin)
            vkCmdFillBuffer(cmd, m_globalMinBuffer, 0, sizeof(uint32_t), 0xFFFFFFFF);

            // Clear debug info
            vkCmdFillBuffer(cmd, m_debugInfoBuffer, 0, sizeof(uint32_t), 0xFFFFFFFF); // debugMinMaxPos
            vkCmdFillBuffer(cmd, m_debugInfoBuffer, sizeof(uint32_t), sizeof(uint32_t), 0); // debugNumRoots

            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            stamp();

            // Pass 1: Register roots (write rootMaxPos[root] via atomicMax)
            recordRegisterRootsPhase(cmd);
            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            // Pass 2: Find global min (read rootMaxPos, atomicMin into globalMin)
            recordFindMinPhase(cmd);
            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            stamp();

            // Pass 3: Resolve (compare rootMaxPos[root] == globalMin, output + advance)
            recordResolvePhase(cmd, peel);
            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            stamp();
        }

        // ─── Composite ───────────────────────────────────────────────────
        recordMemoryBarrier(cmd,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        recordCompositePhase(cmd, swapchainImageIndex);
        stamp();
    }

    // Profiling transfers are AFTER the renderer's final marker and BEFORE
    // ImGui. Their GPU cost is excluded from renderer_gpu_ms but remains in
    // application frame intervals. Keep readback memory private to this slot.
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy counterCopy{0, 0, sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, m_atomicCounterBuffer, profile.readbackBuffer, 1, &counterCopy);
    if (profile.layered) {
        VkBufferCopy debugCopy{0, sizeof(uint32_t), 4 * sizeof(uint32_t)};
        vkCmdCopyBuffer(cmd, m_debugInfoBuffer, profile.readbackBuffer, 1, &debugCopy);
    }
    VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.buffer = profile.readbackBuffer;
    hostBarrier.offset = 0;
    hostBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &hostBarrier, 0, nullptr);
}
// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 1: CLEAR
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordClearPhase(VkCommandBuffer cmd)
{
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

// In recordCollectPhase — update the push constants to include view matrix
// Replace the loop body where push constants are set:

void UFRenderer::recordCollectPhase(VkCommandBuffer cmd,
                                    VkBuffer vertexBuffer, VkBuffer indexBuffer,
                                    const std::vector<SceneObject> &objects,
                                    const glm::mat4 &viewProj,
                                    const glm::mat4 &view)
{
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipeline);

    VkViewport vp{0, (float)m_height, (float)m_width, -(float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Bind set 0: PPLL data
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_collectPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    // Bind set 1: Textures (if available)
    if (m_textureManager)
    {
        VkDescriptorSet texSet = m_textureManager->getDescriptorSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_collectPipelineLayout, 1, 1, &texSet, 0, nullptr);
    }

    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto &obj : objects)
    {
        UFScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = viewProj;
        pc.view = view;
        pc.objectID = obj.objectID;
        pc.nearPlane = nearPlane;
        pc.farPlane = farPlane;
        pc.textureIndex = obj.textureIndex; // -1 = no texture
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

void UFRenderer::recordSortPhase(VkCommandBuffer cmd)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
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

void UFRenderer::recordInitParentsPhase(VkCommandBuffer cmd)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
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

void UFRenderer::recordMergePhase(VkCommandBuffer cmd)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    pc.nearPlane = nearPlane;
    pc.farPlane = farPlane;
    pc.projX = m_projX;
    pc.projY = m_projY;
    pc.colorThreshold = colorThreshold;
    pc.mergeMode = mergeMode;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 6: COMPRESS (Path Compression)
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordCompressPhase(VkCommandBuffer cmd)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void UFRenderer::recordRegisterRootsPhase(VkCommandBuffer cmd)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = 0;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void UFRenderer::recordFindMinPhase(VkCommandBuffer cmd)
{
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_findMinPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 7: RESOLVE (one call per layer)
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex)
{
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
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.nearPlane = nearPlane; // ← NEW
    pc.farPlane = farPlane;   // ← NEW
    pc.projX = m_projX;       // ← NEW
    pc.projY = m_projY;       // ← NEW

    // Debug mode: layer 0 gets the debug code, other layers get normal (will output transparent)
    if (debugDisplayLayer >= 100 && layerIndex == 0)
    {
        pc.targetLayer = static_cast<uint32_t>(debugDisplayLayer);
    }
    else if (debugDisplayLayer >= 100)
    {
        pc.targetLayer = layerIndex; // layers 1-3 will just output transparent in normal mode
    }
    else
    {
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

void UFRenderer::recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex)
{
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

void UFRenderer::recordDebugRegionsPhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex)
{
    VkClearValue clear{};
    clear.color = { {0.02f, 0.02f, 0.08f, 1.0f} };

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = m_compositeRenderPass;
    rp.framebuffer = m_compositeFramebuffers[swapchainImageIndex];
    rp.renderArea = { {0, 0}, {m_width, m_height} };
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_debugRegionsPipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth = m_width;
    pc.screenHeight = m_height;
    pc.targetLayer = static_cast<uint32_t>(debugVisualizationMode);  // visualization mode
    pc.nearPlane = nearPlane;
    pc.farPlane = farPlane;
    pc.projX = m_projX;
    pc.projY = m_projY;
    pc.colorThreshold = colorThreshold;
    pc.mergeMode = mergeMode;
    pc.selectedPixelX = regionSelection.pixelX;
    pc.selectedPixelY = regionSelection.pixelY;
    pc.selectionActive = (regionSelection.active && !bypassUnionFind) ? 1u : 0u;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: MEMORY BARRIER
// ═══════════════════════════════════════════════════════════════════════════════

void UFRenderer::recordMemoryBarrier(VkCommandBuffer cmd,
                                     VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                     VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

bool UFRenderer::createProfilingResources(VulkanContext& context)
{
    const uint32_t frames = SwapchainManager::MAX_FRAMES_IN_FLIGHT;
    m_profileFrames.resize(frames);
    // Four debug uints plus the allocation-request counter for each flight slot.
    for (auto& frame : m_profileFrames) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = 5 * sizeof(uint32_t);
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(context.device, &info, nullptr, &frame.readbackBuffer) != VK_SUCCESS)
            return false;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context.device, frame.readbackBuffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(context.physical_device, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(context.device, &allocation, nullptr, &frame.readbackMemory) != VK_SUCCESS ||
            vkBindBufferMemory(context.device, frame.readbackBuffer, frame.readbackMemory, 0) != VK_SUCCESS ||
            vkMapMemory(context.device, frame.readbackMemory, 0, VK_WHOLE_SIZE, 0, &frame.mapped) != VK_SUCCESS)
            return false;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device, &properties);
    m_timestampPeriod = properties.limits.timestampPeriod;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &count, families.data());
    m_timestampValidBits = families.at(context.graphics_queue_family).timestampValidBits;
    if (m_timestampValidBits && m_timestampPeriod > 0) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = MAX_TIMESTAMPS * frames;
        if (vkCreateQueryPool(context.device, &info, nullptr, &m_timestampQueryPool) != VK_SUCCESS)
            m_timestampQueryPool = VK_NULL_HANDLE; // CPU/diagnostic capture still works.
    }
    return true;
}

void UFRenderer::destroyProfilingResources(VkDevice device)
{
    for (auto& frame : m_profileFrames) {
        if (frame.mapped) vkUnmapMemory(device, frame.readbackMemory);
        if (frame.readbackBuffer) vkDestroyBuffer(device, frame.readbackBuffer, nullptr);
        if (frame.readbackMemory) vkFreeMemory(device, frame.readbackMemory, nullptr);
    }
    m_profileFrames.clear();
    if (m_timestampQueryPool) vkDestroyQueryPool(device, m_timestampQueryPool, nullptr);
    m_timestampQueryPool = VK_NULL_HANDLE;
}

std::optional<performance::GpuFrameResult> UFRenderer::resolveCompletedFrame(VkDevice device, uint32_t slot)
{
    auto& frame = m_profileFrames.at(slot);
    if (!frame.pending) return std::nullopt;
    frame.pending = false;
    performance::GpuFrameResult result;
    result.frameId = frame.frameId;
    // The application has observed this generation's fence, so both these
    // coherent transfers and the query reset/write commands have executed.
    const auto* data = static_cast<const uint32_t*>(frame.mapped);
    const uint64_t requests = data[0];
    const uint64_t capacity = frame.nodeCapacity;
    result.allocationRequests = requests;
    result.storedFragments = std::min(requests, capacity);
    result.nodePoolOverflow = requests > capacity ? requests - capacity : 0;
    // Retained != stored: uf_sort truncates each pixel's list to 32 nodes.
    // No existing shader counter measures that truncation. Leave both NA.
    debugStats.fragmentCount = static_cast<uint32_t>(requests);
    debugStats.nodeCapacity = frame.nodeCapacity;
    debugStats.overflowed = requests > capacity; // Exactly full is NOT overflow.
    debugStats.usage = capacity ? float(100.0 * *result.storedFragments / capacity) : 0.0f;
    if (frame.layered) {
        debugGPUInfo.minMaxPos = data[1];
        debugGPUInfo.numRegisteredRoots = data[2]; // Last peel, not peel zero.
    }
    m_timingReady = false;
    if (!frame.timed) {
        result.status = frame.evaluationExport ? "evaluation_export_excluded" :
            !frame.layered ? "debug_visualization_not_measured" :
            timestampsSupported() ? "timing_disabled" : "timestamps_unsupported";
        return result;
    }
    struct Query { uint64_t ticks = 0, available = 0; };
    std::array<Query, MAX_TIMESTAMPS> queries{};
    const VkResult queryResult = vkGetQueryPoolResults(device, m_timestampQueryPool,
        slot * MAX_TIMESTAMPS, frame.timestampCount,
        sizeof(Query) * frame.timestampCount, queries.data(), sizeof(Query),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (queryResult != VK_SUCCESS && queryResult != VK_NOT_READY) {
        result.status = "query_error_" + std::to_string(static_cast<int>(queryResult));
        return result;
    }
    auto ms = [&](uint32_t a, uint32_t b) -> std::optional<double> {
        if (a >= frame.timestampCount || b >= frame.timestampCount ||
            !queries[a].available || !queries[b].available) return std::nullopt;
        return performance::timestampMilliseconds(queries[a].ticks, queries[b].ticks,
                                                  m_timestampValidBits, m_timestampPeriod);
    };
    result.clearMs = ms(0, 1);
    result.collectMs = ms(1, 2);
    result.sortMs = ms(2, 3);
    result.initParentsMs = ms(3, 4);
    if (frame.grouping) result.groupingMs = ms(4, 5);
    result.rendererMs = ms(0, frame.timestampCount - 1);
    if (frame.layered) {
        const uint32_t finalResolve = 5 + 3 * NUM_LAYERS;
        result.extractionMs = ms(5, finalResolve);
        result.compositeMs = ms(finalResolve, finalResolve + 1);
    }
    const bool allAvailable = std::all_of(queries.begin(), queries.begin() + frame.timestampCount,
                                         [](const Query& q) { return q.available != 0; });
    result.status = allAvailable ? (frame.grouping ? "available" : "grouping_bypassed")
                                 : "query_unavailable_after_fence";
    // UI fields are refreshed only from this completed frame. CSV never reads
    // these display values and never substitutes them for missing samples.
    timingResults = {};
    timingResults.clear = float(result.clearMs.value_or(0));
    timingResults.collect = float(result.collectMs.value_or(0));
    timingResults.sort = float(result.sortMs.value_or(0));
    timingResults.initParents = float(result.initParentsMs.value_or(0));
    timingResults.grouping = float(result.groupingMs.value_or(0));
    timingResults.total = float(result.rendererMs.value_or(0));
    if (frame.layered) {
        for (uint32_t i = 0; i < NUM_LAYERS; ++i) {
            const uint32_t start = 5 + i * 3;
            auto& peel = timingResults.peels[i];
            peel.clearTable = float(ms(start, start + 1).value_or(0));
            peel.registerRoots = float(ms(start + 1, start + 2).value_or(0)); // Includes find-min.
            peel.resolve = float(ms(start + 2, start + 3).value_or(0));
            peel.total = float(ms(start, start + 3).value_or(0));
        }
        timingResults.peelTotal = float(result.extractionMs.value_or(0));
        timingResults.composite = float(result.compositeMs.value_or(0));
    }
    m_timingReady = allAvailable;
    return result;
}

void UFRenderer::selectRegionAtPixel(int32_t x, int32_t y)
{
    if (x < 0 || y < 0 ||
        static_cast<uint32_t>(x) >= m_width ||
        static_cast<uint32_t>(y) >= m_height)
    {
        clearRegionSelection();
        return;
    }

    regionSelection.pixelX = x;
    regionSelection.pixelY = y;
    regionSelection.active = true;
}

bool UFRenderer::requestEvaluationExport(std::string filename)
{
    if (isEvaluationExportPending()) return false;
    if (m_performanceCaptureActive) {
        m_evaluationStatus = "Stop or cancel performance capture before exporting fragment evaluation.";
        return false;
    }
    if (bypassUnionFind) {
        m_evaluationStatus = "Enable Union-Find before exporting.";
        return false;
    }
    if (filename.empty()) {
        m_evaluationStatus = "Evaluation filename is empty.";
        return false;
    }
    m_evaluationPath = std::move(filename);
    m_evaluationStatus = "Evaluation export requested.";
    return true;
}

void UFRenderer::destroyEvaluationReadback(VkDevice device)
{
    if (m_evaluationReadback.buffer)
        vkDestroyBuffer(device, m_evaluationReadback.buffer, nullptr);
    if (m_evaluationReadback.memory)
        vkFreeMemory(device, m_evaluationReadback.memory, nullptr);
    m_evaluationReadback = {};
}

bool UFRenderer::prepareEvaluationReadback(VulkanContext& context)
{
    auto& snapshot = m_evaluationReadback;
    try {
        if (bypassUnionFind || m_nodeCapacity == 0 || m_width == 0 || m_height == 0)
            throw std::runtime_error("No active Union-Find workload");
        destroyEvaluationReadback(context.device);

        static_assert(sizeof(UFFragmentNode) == 32, "GPU node layout mismatch");
        const VkDeviceSize pixels = VkDeviceSize(m_width) * m_height;
        snapshot.headsOffset = sizeof(uint32_t);
        snapshot.parentsOffset =
            snapshot.headsOffset + pixels * sizeof(uint32_t);
        snapshot.nodesOffset = snapshot.parentsOffset
            + VkDeviceSize(m_nodeCapacity) * sizeof(uint32_t);
        snapshot.bytes = snapshot.nodesOffset
            + VkDeviceSize(m_nodeCapacity) * sizeof(UFFragmentNode);

        const auto requireSuccess = [](VkResult result, const char* operation) {
            if (result != VK_SUCCESS)
                throw std::runtime_error(
                    std::string(operation) + " failed (" +
                    std::to_string(static_cast<int>(result)) + ")");
        };

        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = snapshot.bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        requireSuccess(vkCreateBuffer(
            context.device, &info, nullptr, &snapshot.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
            context.device, snapshot.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(context.physical_device, &properties);
        int bestScore = -1;
        uint32_t selectedType = 0;
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
            const auto flags = properties.memoryTypes[i].propertyFlags;
            if (!(requirements.memoryTypeBits & (1u << i)) ||
                !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                continue;
            const int score =
                ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 16 : 0) +
                (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 4 : 0) +
                ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 2 : 0);
            if (score > bestScore) {
                bestScore = score;
                selectedType = i;
            }
        }
        if (bestScore < 0)
            throw std::runtime_error("No host-visible readback memory type");
        snapshot.coherent =
            (properties.memoryTypes[selectedType].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = selectedType;
        requireSuccess(vkAllocateMemory(
            context.device, &allocation, nullptr, &snapshot.memory),
            "vkAllocateMemory");
        requireSuccess(vkBindBufferMemory(
            context.device, snapshot.buffer, snapshot.memory, 0),
            "vkBindBufferMemory");

        snapshot.metadata.width = m_width;
        snapshot.metadata.height = m_height;
        snapshot.metadata.node_capacity = m_nodeCapacity;
        snapshot.metadata.per_pixel_limit = MAX_FRAGS_PER_PIXEL;
        snapshot.metadata.merge_mode = mergeMode;
        snapshot.metadata.proximity_scale = proximityScale;
        snapshot.metadata.plane_threshold = planeThreshold;
        snapshot.metadata.color_threshold = colorThreshold;
        return true;
    }
    catch (const std::exception& error) {
        destroyEvaluationReadback(context.device);
        m_evaluationPath.clear();
        m_evaluationStatus = std::string("Evaluation export failed: ") + error.what();
        return false;
    }
}

void UFRenderer::recordEvaluationReadback(
    VkCommandBuffer cmd, uint32_t mergeIterationCount)
{
    auto& snapshot = m_evaluationReadback;
    snapshot.metadata.merge_iterations = mergeIterationCount;

    // Include prior transfer clears as well as shader writes. Empty pixels may
    // still contain the value produced by the transfer clear.
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    VkBufferCopy counterCopy{0, 0, sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, m_atomicCounterBuffer, snapshot.buffer, 1, &counterCopy);

    VkBufferImageCopy headsCopy{};
    headsCopy.bufferOffset = snapshot.headsOffset;
    headsCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    headsCopy.imageExtent = {m_width, m_height, 1};
    vkCmdCopyImageToBuffer(
        cmd, m_headPointerImage, VK_IMAGE_LAYOUT_GENERAL,
        snapshot.buffer, 1, &headsCopy);

    VkBufferCopy parentsCopy{
        0, snapshot.parentsOffset,
        VkDeviceSize(m_nodeCapacity) * sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, m_parentBuffer, snapshot.buffer, 1, &parentsCopy);
    VkBufferCopy nodesCopy{
        0, snapshot.nodesOffset,
        VkDeviceSize(m_nodeCapacity) * sizeof(UFFragmentNode)};
    vkCmdCopyBuffer(cmd, m_nodeBuffer, snapshot.buffer, 1, &nodesCopy);

    // Readback writes become visible to the host after the submitted fence.
    // Extraction and subsequent clears must also follow the snapshot reads.
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_TRANSFER_WRITE_BIT);
    snapshot.recorded = true;
}

bool UFRenderer::finishEvaluationExport(VkDevice device, VkFence submittedFence)
{
    auto& snapshot = m_evaluationReadback;
    if (!snapshot.recorded) return false;

    // A failed fence wait is a rendering error: do not free resources that may
    // still be in flight and then continue rendering.
    if (vkWaitForFences(device, 1, &submittedFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("Failed to wait for evaluation capture");

    void* mapped = nullptr;
    try {
        if (vkMapMemory(device, snapshot.memory, 0, VK_WHOLE_SIZE, 0, &mapped)
            != VK_SUCCESS)
            throw std::runtime_error("Could not map evaluation readback");
        if (!snapshot.coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = snapshot.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS)
                throw std::runtime_error("Could not invalidate evaluation memory");
        }

        const auto* bytes = static_cast<const std::byte*>(mapped);
        uint32_t requests = 0;
        std::memcpy(&requests, bytes, sizeof(requests));
        snapshot.metadata.allocation_requests = requests;
        const uint32_t stored = std::min(
            requests, snapshot.metadata.node_capacity);
        const auto* heads = reinterpret_cast<const uint32_t*>(
            bytes + snapshot.headsOffset);
        const auto* parents = reinterpret_cast<const uint32_t*>(
            bytes + snapshot.parentsOffset);
        const auto* nodes = reinterpret_cast<const UFFragmentNode*>(
            bytes + snapshot.nodesOffset);
        const auto analysis = uf_eval::analyze<UFFragmentNode>(
            std::span<const UFFragmentNode>(nodes, stored),
            std::span<const uint32_t>(parents, stored),
            std::span<const uint32_t>(
                heads, std::size_t(snapshot.metadata.width) *
                    snapshot.metadata.height),
            snapshot.metadata.per_pixel_limit);

        const auto destination = std::filesystem::absolute(m_evaluationPath);
        uf_eval::write_csv(destination, analysis, snapshot.metadata);
        m_evaluationStatus = "Saved " + destination.string() + " (" +
            std::to_string(analysis.retained) + " retained fragments).";
    }
    catch (const std::exception& error) {
        m_evaluationStatus =
            std::string("Evaluation export failed: ") + error.what();
    }
    if (mapped) vkUnmapMemory(device, snapshot.memory);
    destroyEvaluationReadback(device);
    m_evaluationPath.clear();
    std::cout << "[UF evaluation] " << m_evaluationStatus << '\n';
    return true; // An export frame was processed, including a failed CSV write.
}