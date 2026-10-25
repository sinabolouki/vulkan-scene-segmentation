#include "sam2_renderer.h"

#include "vulkan_context.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "mesh.h"
#include "texture_manager.h"
#include "uf_push_constants.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

// =========================================================================
// init / destroy
// =========================================================================

bool Sam2Renderer::init(VulkanContext& ctx, const Config& cfg)
{
    m_cfg    = cfg;
    m_width  = cfg.width;
    m_height = cfg.height;

    if (!createDataStructures(ctx))         return false;
    if (!createLayerImages(ctx))            return false;
    if (!createRenderPasses(ctx.device))    return false;
    if (!createFramebuffers(ctx.device))    return false;
    if (!createSampler(ctx.device))         return false;
    if (!createDescriptorSetLayout(ctx.device)) return false;
    if (!createDescriptorPool(ctx.device))  return false;
    if (!createDescriptorSet(ctx.device))   return false;
    if (!createPipelineLayouts(ctx.device)) return false;
    if (!createPipelines(ctx.device))       return false;
    if (!createStagingBuffer(ctx))          return false;
    if (!createCommandResources(ctx))       return false;

    std::cout << "[Sam2] Initialized " << m_width << "x" << m_height
              << ", maxLayers=" << m_cfg.maxLayers
              << ", nodeCapacity=" << m_nodeCapacity << std::endl;
    return true;
}

// =========================================================================
// Resource creation
// =========================================================================

bool Sam2Renderer::createDataStructures(VulkanContext& ctx)
{
    // ---- Head pointer image (R32_UINT, storage image) ---------------
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format    = VK_FORMAT_R32_UINT;
        info.extent    = {m_width, m_height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples   = VK_SAMPLE_COUNT_1_BIT;
        info.tiling    = VK_IMAGE_TILING_OPTIMAL;
        info.usage     = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(ctx.device, &info, nullptr, &m_headPointerImage) != VK_SUCCESS)
            return false;

        VkMemoryRequirements req; vkGetImageMemoryRequirements(ctx.device, m_headPointerImage, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = findMemoryType(ctx.physical_device, req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(ctx.device, &ai, nullptr, &m_headPointerMemory) != VK_SUCCESS) return false;
        vkBindImageMemory(ctx.device, m_headPointerImage, m_headPointerMemory, 0);

        VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        v.image    = m_headPointerImage;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D;
        v.format   = VK_FORMAT_R32_UINT;
        v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx.device, &v, nullptr, &m_headPointerView) != VK_SUCCESS) return false;
    }

    // ---- Node buffer (SSBO) -----------------------------------------
    m_nodeCapacity = m_width * m_height * UFRenderer::AVG_FRAGS_PER_PIXEL;
    if (!createBuffer(ctx,
                      m_nodeCapacity * sizeof(UFFragmentNode),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      m_nodeBuffer, m_nodeBufferMemory))
        return false;

    // ---- Atomic counter (SSBO) --------------------------------------
    if (!createBuffer(ctx,
                      sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      m_atomicCounterBuffer, m_atomicCounterMemory))
        return false;

    // ---- Union-find parent buffer (SSBO, one entry per node) ----------
    if (!createBuffer(ctx,
                      m_nodeCapacity * sizeof(uint32_t),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      m_parentBuffer, m_parentBufferMemory))
        return false;

    // ---- Region labels of one depth layer (SSBO, read back) -----------
    m_labelBufferSize = VkDeviceSize(m_width) * m_height * sizeof(uint32_t);
    if (!createBuffer(ctx,
                      m_labelBufferSize,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      m_regionLabelBuffer, m_regionLabelMemory))
        return false;

    return true;
}

bool Sam2Renderer::createLayerImages(VulkanContext& ctx)
{
    m_layerImages.resize(m_cfg.maxLayers);
    for (uint32_t i = 0; i < m_cfg.maxLayers; ++i)
    {
        if (!createImage(ctx, m_width, m_height, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                       | VK_IMAGE_USAGE_SAMPLED_BIT
                       | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         m_layerImages[i].image,
                         m_layerImages[i].memory,
                         m_layerImages[i].view))
            return false;
    }
    return true;
}

bool Sam2Renderer::createFramebuffers(VkDevice device)
{
    // No-attachment framebuffer
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_noAttachmentRenderPass;
        fb.attachmentCount = 0;
        fb.width  = m_width;
        fb.height = m_height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_noAttachmentFramebuffer) != VK_SUCCESS)
            return false;
    }
    // Layer framebuffers
    for (uint32_t i = 0; i < m_cfg.maxLayers; ++i)
    {
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_layerRenderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = &m_layerImages[i].view;
        fb.width  = m_width;
        fb.height = m_height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_layerImages[i].framebuffer) != VK_SUCCESS)
            return false;
    }
    return true;
}

bool Sam2Renderer::createSampler(VkDevice device)
{
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.maxLod = 1.0f;
    return vkCreateSampler(device, &info, nullptr, &m_sampler) == VK_SUCCESS;
}

bool Sam2Renderer::createDescriptorSetLayout(VkDevice device)
{
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = (uint32_t)b.size();
    info.pBindings    = b.data();
    return vkCreateDescriptorSetLayout(device, &info, nullptr, &m_ppllDescLayout) == VK_SUCCESS;
}

bool Sam2Renderer::createDescriptorPool(VkDevice device)
{
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1};
    sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};

    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount = (uint32_t)sizes.size();
    info.pPoolSizes    = sizes.data();
    info.maxSets = 1;
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool Sam2Renderer::createDescriptorSet(VkDevice device)
{
    VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    a.descriptorPool     = m_descriptorPool;
    a.descriptorSetCount = 1;
    a.pSetLayouts        = &m_ppllDescLayout;
    if (vkAllocateDescriptorSets(device, &a, &m_ppllDescSet) != VK_SUCCESS) return false;

    VkDescriptorImageInfo headI{};
    headI.imageView   = m_headPointerView;
    headI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo nodeI   {m_nodeBuffer,          0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo counterI{m_atomicCounterBuffer, 0, sizeof(uint32_t)};
    VkDescriptorBufferInfo parentI {m_parentBuffer,        0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo labelI  {m_regionLabelBuffer,   0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 5> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_ppllDescSet; w[0].dstBinding = 0;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].descriptorCount = 1; w[0].pImageInfo = &headI;

    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_ppllDescSet; w[1].dstBinding = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].descriptorCount = 1; w[1].pBufferInfo = &nodeI;

    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_ppllDescSet; w[2].dstBinding = 2;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].descriptorCount = 1; w[2].pBufferInfo = &counterI;

    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_ppllDescSet; w[3].dstBinding = 3;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].descriptorCount = 1; w[3].pBufferInfo = &parentI;

    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_ppllDescSet; w[4].dstBinding = 4;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[4].descriptorCount = 1; w[4].pBufferInfo = &labelI;

    vkUpdateDescriptorSets(device, (uint32_t)w.size(), w.data(), 0, nullptr);
    return true;
}

bool Sam2Renderer::createPipelineLayouts(VkDevice device)
{
    // ---- Collect layout: set 0 = PPLL, set 1 = textures --------------
    {
        std::vector<VkDescriptorSetLayout> sets;
        sets.push_back(m_ppllDescLayout);
        if (m_textureManager) sets.push_back(m_textureManager->getDescriptorSetLayout());

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFScenePushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = (uint32_t)sets.size();
        info.pSetLayouts    = sets.data();
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges    = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_collectPipelineLayout) != VK_SUCCESS)
            return false;
    }
    // ---- Fullscreen layout: set 0 = PPLL only ------------------------
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        range.size = sizeof(UFFullscreenPushConstants);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1;
        info.pSetLayouts    = &m_ppllDescLayout;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges    = &range;
        if (vkCreatePipelineLayout(device, &info, nullptr, &m_fullscreenPipelineLayout) != VK_SUCCESS)
            return false;
    }
    return true;
}

bool Sam2Renderer::createPipelines(VkDevice device)
{
    // Collect and the union-find passes use the UNION_FIND renderer's shaders, so the
    // captured fragments and regions match what that renderer computes.
    const std::string sd = "shaders/sam2";
    const std::string ufd = "shaders/union_find";

    VkShaderModule collectVert   = ShaderUtils::createShaderModule(device, ufd + "/uf_collect.vert.spv");
    VkShaderModule collectFrag   = ShaderUtils::createShaderModule(device, ufd + "/uf_collect.frag.spv");
    VkShaderModule fullscreenVert= ShaderUtils::createShaderModule(device, sd + "/sam2_fullscreen.vert.spv");
    VkShaderModule sortFrag      = ShaderUtils::createShaderModule(device, sd + "/sam2_sort.frag.spv");
    VkShaderModule resolveFrag   = ShaderUtils::createShaderModule(device, sd + "/sam2_resolve.frag.spv");
    VkShaderModule initFrag      = ShaderUtils::createShaderModule(device, ufd + "/uf_init_parents.frag.spv");
    VkShaderModule mergeFrag     = ShaderUtils::createShaderModule(device, ufd + "/uf_merge.frag.spv");
    VkShaderModule compressFrag  = ShaderUtils::createShaderModule(device, ufd + "/uf_compress.frag.spv");
    VkShaderModule labelsFrag    = ShaderUtils::createShaderModule(device, sd + "/sam2_labels.frag.spv");

    if (!collectVert || !collectFrag || !fullscreenVert || !sortFrag || !resolveFrag ||
        !initFrag || !mergeFrag || !compressFrag || !labelsFrag)
    {
        std::cerr << "[Sam2] Failed to load shaders" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription vtxBind = Vertex::getBindingDescription();
    auto vtxAttr = Vertex::getAttributeDescriptions();

    PipelineBuilder b;

    // ---- Collect ----
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
    if (!m_collectPipeline) return false;

    // ---- Sort ----
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
    if (!m_sortPipeline) return false;

    // ---- Resolve ----
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
    if (!m_resolvePipeline) return false;

    // ---- Union-find + region labels (no attachments) ----
    auto buildNoAttachment = [&](VkShaderModule frag)
    {
        b.clear();
        return b
            .setVertexShader(fullscreenVert)
            .setFragmentShader(frag)
            .setNoVertexInput()
            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
            .disableDepthTest()
            .setNoColorAttachments()
            .setPipelineLayout(m_fullscreenPipelineLayout)
            .setRenderPass(m_noAttachmentRenderPass, 0)
            .build(device);
    };
    m_initParentsPipeline  = buildNoAttachment(initFrag);
    m_mergePipeline        = buildNoAttachment(mergeFrag);
    m_compressPipeline     = buildNoAttachment(compressFrag);
    m_regionLabelsPipeline = buildNoAttachment(labelsFrag);
    if (!m_initParentsPipeline || !m_mergePipeline || !m_compressPipeline || !m_regionLabelsPipeline)
        return false;

    vkDestroyShaderModule(device, collectVert,   nullptr);
    vkDestroyShaderModule(device, collectFrag,   nullptr);
    vkDestroyShaderModule(device, fullscreenVert,nullptr);
    vkDestroyShaderModule(device, sortFrag,      nullptr);
    vkDestroyShaderModule(device, resolveFrag,   nullptr);
    vkDestroyShaderModule(device, initFrag,      nullptr);
    vkDestroyShaderModule(device, mergeFrag,     nullptr);
    vkDestroyShaderModule(device, compressFrag,  nullptr);
    vkDestroyShaderModule(device, labelsFrag,    nullptr);
    return true;
}

bool Sam2Renderer::createStagingBuffer(VulkanContext& ctx)
{
    m_stagingBufferSize = VkDeviceSize(m_width) * m_height * 4;

    if (!createBuffer(ctx, m_stagingBufferSize,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_stagingBuffer, m_stagingMemory))
        return false;

    if (vkMapMemory(ctx.device, m_stagingMemory, 0, m_stagingBufferSize, 0, &m_stagingMapped) != VK_SUCCESS)
        return false;

    if (!createBuffer(ctx, m_labelBufferSize,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_labelStagingBuffer, m_labelStagingMemory))
        return false;

    if (vkMapMemory(ctx.device, m_labelStagingMemory, 0, m_labelBufferSize, 0, &m_labelStagingMapped) != VK_SUCCESS)
        return false;
    return true;
}

bool Sam2Renderer::createCommandResources(VulkanContext& ctx)
{
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = ctx.graphics_queue_family;
    if (vkCreateCommandPool(ctx.device, &pi, nullptr, &m_captureCmdPool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = m_captureCmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device, &ai, &m_captureCmd) != VK_SUCCESS) return false;

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    return vkCreateFence(ctx.device, &fi, nullptr, &m_captureFence) == VK_SUCCESS;
}

// =========================================================================
// Small helpers
// =========================================================================

uint32_t Sam2Renderer::findMemoryType(VkPhysicalDevice pd, uint32_t typeBits,
                                      VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    std::cerr << "[Sam2] No suitable memory type" << std::endl;
    return 0;
}

bool Sam2Renderer::createImage(VulkanContext& ctx, uint32_t w, uint32_t h,
                               VkFormat format, VkImageUsageFlags usage,
                               VkImageAspectFlags aspect,
                               VkImage& image, VkDeviceMemory& memory, VkImageView& view)
{
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType     = VK_IMAGE_TYPE_2D;
    ii.format        = format;
    ii.extent        = {w, h, 1};
    ii.mipLevels     = 1;
    ii.arrayLayers   = 1;
    ii.samples       = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ii.usage         = usage;
    ii.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(ctx.device, &ii, nullptr, &image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx.physical_device, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(ctx.device, &ai, nullptr, &memory) != VK_SUCCESS) return false;
    vkBindImageMemory(ctx.device, image, memory, 0);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = format;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    return vkCreateImageView(ctx.device, &vi, nullptr, &view) == VK_SUCCESS;
}

bool Sam2Renderer::createBuffer(VulkanContext& ctx, VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer& buffer, VkDeviceMemory& memory)
{
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size        = size;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device, &bi, nullptr, &buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx.physical_device, req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx.device, &ai, nullptr, &memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(ctx.device, buffer, memory, 0);
    return true;
}

void Sam2Renderer::recordMemoryBarrier(VkCommandBuffer cmd,
                                       VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void Sam2Renderer::recordCollectPhase(VkCommandBuffer cmd,
                                      VkBuffer vertexBuffer, VkBuffer indexBuffer,
                                      const std::vector<SceneObject>& objects,
                                      const glm::mat4& viewProj,
                                      const glm::mat4& view)
{
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass  = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea  = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipeline);

    // Flip Y like UFRenderer::recordCollectPhase does.
    VkViewport vp{0.0f, (float)m_height, (float)m_width, -(float)m_height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Set 0: PPLL
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_collectPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    // Set 1: textures (if attached)
    if (m_textureManager)
    {
        VkDescriptorSet texSet = m_textureManager->getDescriptorSet();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_collectPipelineLayout, 1, 1, &texSet, 0, nullptr);
    }

    VkBuffer     vbs[]   = {vertexBuffer};
    VkDeviceSize offs[]  = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto& obj : objects)
    {
        UFScenePushConstants pc{};
        pc.model        = glm::mat4(1.0f);
        pc.viewProj     = viewProj;
        pc.view         = view;
        pc.objectID     = obj.objectID;
        pc.nearPlane    = nearPlane;
        pc.farPlane     = farPlane;
        pc.textureIndex = obj.textureIndex;
        vkCmdPushConstants(cmd, m_collectPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

void Sam2Renderer::recordSortPhase(VkCommandBuffer cmd)
{
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass  = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea  = { {0, 0}, {m_width, m_height} };
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
    pc.screenWidth    = m_width;
    pc.screenHeight   = m_height;
    pc.targetLayer    = 0;
    pc.nearPlane      = nearPlane;
    pc.farPlane       = farPlane;
    pc.projX          = m_projX;
    pc.projY          = m_projY;
    pc.colorThreshold = colorThreshold;
    pc.mergeMode      = mergeMode;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void Sam2Renderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex)
{
    VkClearValue clear{};
    clear.color = { {0.0f, 0.0f, 0.0f, 0.0f} };

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass      = m_layerRenderPass;
    rp.framebuffer     = m_layerImages[layerIndex].framebuffer;
    rp.renderArea      = { {0, 0}, {m_width, m_height} };
    rp.clearValueCount = 1;
    rp.pClearValues    = &clear;
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
    pc.screenWidth    = m_width;
    pc.screenHeight   = m_height;
    pc.targetLayer    = layerIndex;
    pc.nearPlane      = nearPlane;
    pc.farPlane       = farPlane;
    pc.projX          = m_projX;
    pc.projY          = m_projY;
    pc.colorThreshold = colorThreshold;
    pc.mergeMode      = mergeMode;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    // After this, layer image is in SHADER_READ_ONLY_OPTIMAL (final layout).
}

void Sam2Renderer::recordFullscreenPass(VkCommandBuffer cmd, VkPipeline pipeline, uint32_t targetLayer)
{
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass  = m_noAttachmentRenderPass;
    rp.framebuffer = m_noAttachmentFramebuffer;
    rp.renderArea  = { {0, 0}, {m_width, m_height} };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkViewport vp{0, 0, (float)m_width, (float)m_height, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {m_width, m_height} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_fullscreenPipelineLayout, 0, 1, &m_ppllDescSet, 0, nullptr);

    UFFullscreenPushConstants pc{};
    pc.planeThreshold = planeThreshold;
    pc.proximityScale = proximityScale;
    pc.screenWidth    = m_width;
    pc.screenHeight   = m_height;
    pc.targetLayer    = targetLayer;
    pc.nearPlane      = nearPlane;
    pc.farPlane       = farPlane;
    pc.projX          = m_projX;
    pc.projY          = m_projY;
    pc.colorThreshold = colorThreshold;
    pc.mergeMode      = mergeMode;
    vkCmdPushConstants(cmd, m_fullscreenPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void Sam2Renderer::recordUnionFindPhase(VkCommandBuffer cmd)
{
    // Same sequence as UFRenderer::recordCommands: init, then (merge, compress) x N.
    auto shaderBarrier = [&]()
    {
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };

    recordFullscreenPass(cmd, m_initParentsPipeline, 0);
    shaderBarrier();

    for (uint32_t iter = 0; iter < mergeIterations; ++iter)
    {
        recordFullscreenPass(cmd, m_mergePipeline, 0);
        shaderBarrier();
        recordFullscreenPass(cmd, m_compressPipeline, 0);
        shaderBarrier();
    }
}

void Sam2Renderer::recordCopyRegionLabelsToStaging(VkCommandBuffer cmd)
{
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    VkBufferCopy copy{};
    copy.size = m_labelBufferSize;
    vkCmdCopyBuffer(cmd, m_regionLabelBuffer, m_labelStagingBuffer, 1, &copy);

    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
}

void Sam2Renderer::recordCopyLayerToStaging(VkCommandBuffer cmd, uint32_t layerIndex)
{
    VkImage img = m_layerImages[layerIndex].image;

    // SHADER_READ_ONLY_OPTIMAL -> TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ib.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image             = img;
    ib.subresourceRange  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &ib);

    // Copy image -> staging buffer (tightly packed)
    VkBufferImageCopy c{};
    c.bufferOffset      = 0;
    c.bufferRowLength   = 0;
    c.bufferImageHeight = 0;
    c.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.mipLevel       = 0;
    c.imageSubresource.baseArrayLayer = 0;
    c.imageSubresource.layerCount     = 1;
    c.imageOffset = {0, 0, 0};
    c.imageExtent = {m_width, m_height, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_stagingBuffer, 1, &c);

    // Transition back to SHADER_READ_ONLY_OPTIMAL (so ImGui can still sample it later)
    ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    ib.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &ib);

    // Ensure the buffer write is visible to the host after the fence.
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer              = m_stagingBuffer;
    bb.offset              = 0;
    bb.size                = m_stagingBufferSize;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bb, 0, nullptr);
}

// =========================================================================
// Public: real-time preview (layer 0 only)
// =========================================================================

void Sam2Renderer::recordPreview(VkCommandBuffer cmd,
                                 VkBuffer vertexBuffer,
                                 VkBuffer indexBuffer,
                                 const std::vector<SceneObject>& objects,
                                 const glm::mat4& viewProj,
                                 const glm::mat4& view,
                                 const glm::mat4& projection)
{
    m_projX = projection[0][0];
    m_projY = projection[1][1];

    recordClearPhase(cmd);

    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj, view);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    recordSortPhase(cmd);
    recordMemoryBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // Resolve only layer 0 for the preview
    recordResolvePhase(cmd, 0);
    // Layer 0 image is now SHADER_READ_ONLY_OPTIMAL — ImGui can sample it.
}

void Sam2Renderer::destroy(VkDevice device)
{
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);

    if (m_captureFence)   { vkDestroyFence(device, m_captureFence, nullptr);         m_captureFence   = VK_NULL_HANDLE; }
    if (m_captureCmdPool) { vkDestroyCommandPool(device, m_captureCmdPool, nullptr); m_captureCmdPool = VK_NULL_HANDLE; }
    m_captureCmd = VK_NULL_HANDLE; // freed with the pool

    if (m_collectPipeline) { vkDestroyPipeline(device, m_collectPipeline, nullptr); m_collectPipeline = VK_NULL_HANDLE; }
    if (m_sortPipeline)    { vkDestroyPipeline(device, m_sortPipeline,    nullptr); m_sortPipeline    = VK_NULL_HANDLE; }
    if (m_resolvePipeline) { vkDestroyPipeline(device, m_resolvePipeline, nullptr); m_resolvePipeline = VK_NULL_HANDLE; }
    if (m_initParentsPipeline)  { vkDestroyPipeline(device, m_initParentsPipeline,  nullptr); m_initParentsPipeline  = VK_NULL_HANDLE; }
    if (m_mergePipeline)        { vkDestroyPipeline(device, m_mergePipeline,        nullptr); m_mergePipeline        = VK_NULL_HANDLE; }
    if (m_compressPipeline)     { vkDestroyPipeline(device, m_compressPipeline,     nullptr); m_compressPipeline     = VK_NULL_HANDLE; }
    if (m_regionLabelsPipeline) { vkDestroyPipeline(device, m_regionLabelsPipeline, nullptr); m_regionLabelsPipeline = VK_NULL_HANDLE; }

    if (m_collectPipelineLayout)    { vkDestroyPipelineLayout(device, m_collectPipelineLayout,    nullptr); m_collectPipelineLayout    = VK_NULL_HANDLE; }
    if (m_fullscreenPipelineLayout) { vkDestroyPipelineLayout(device, m_fullscreenPipelineLayout, nullptr); m_fullscreenPipelineLayout = VK_NULL_HANDLE; }

    // Pool, staging, images, framebuffers, PPLL buffers
    destroySizeDependent(device);

    if (m_ppllDescLayout) { vkDestroyDescriptorSetLayout(device, m_ppllDescLayout, nullptr); m_ppllDescLayout = VK_NULL_HANDLE; }
    if (m_sampler)        { vkDestroySampler(device, m_sampler, nullptr); m_sampler = VK_NULL_HANDLE; }

    if (m_noAttachmentRenderPass) { vkDestroyRenderPass(device, m_noAttachmentRenderPass, nullptr); m_noAttachmentRenderPass = VK_NULL_HANDLE; }
    if (m_layerRenderPass)        { vkDestroyRenderPass(device, m_layerRenderPass,        nullptr); m_layerRenderPass        = VK_NULL_HANDLE; }
}


void Sam2Renderer::recreate(VulkanContext& ctx, uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == 0 || newHeight == 0) return; // minimized window

    vkDeviceWaitIdle(ctx.device);
    destroySizeDependent(ctx.device);

    m_width      = newWidth;
    m_height     = newHeight;
    m_cfg.width  = newWidth;
    m_cfg.height = newHeight;

    // Render passes, pipelines, set layout, sampler, command pool are size-independent → kept.
    if (!createDataStructures(ctx))      throw std::runtime_error("[Sam2] recreate: data structures");
    if (!createLayerImages(ctx))         throw std::runtime_error("[Sam2] recreate: layer images");
    if (!createFramebuffers(ctx.device)) throw std::runtime_error("[Sam2] recreate: framebuffers");
    if (!createStagingBuffer(ctx))       throw std::runtime_error("[Sam2] recreate: staging buffer");
    if (!createDescriptorPool(ctx.device)) throw std::runtime_error("[Sam2] recreate: descriptor pool");
    if (!createDescriptorSet(ctx.device))  throw std::runtime_error("[Sam2] recreate: descriptor set");

    std::cout << "[Sam2] Recreated " << m_width << "x" << m_height
              << ", nodeCapacity=" << m_nodeCapacity << std::endl;
}

void Sam2Renderer::destroySizeDependent(VkDevice device)
{
    // Descriptor pool (the set is freed with it; the set LAYOUT is kept)
    if (m_descriptorPool) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        m_ppllDescSet    = VK_NULL_HANDLE;
    }

    // Staging buffer
    if (m_stagingMapped) { vkUnmapMemory(device, m_stagingMemory); m_stagingMapped = nullptr; }
    if (m_stagingBuffer) { vkDestroyBuffer(device, m_stagingBuffer, nullptr); m_stagingBuffer = VK_NULL_HANDLE; }
    if (m_stagingMemory) { vkFreeMemory(device, m_stagingMemory, nullptr);    m_stagingMemory = VK_NULL_HANDLE; }
    m_stagingBufferSize = 0;

    if (m_labelStagingMapped) { vkUnmapMemory(device, m_labelStagingMemory); m_labelStagingMapped = nullptr; }
    if (m_labelStagingBuffer) { vkDestroyBuffer(device, m_labelStagingBuffer, nullptr); m_labelStagingBuffer = VK_NULL_HANDLE; }
    if (m_labelStagingMemory) { vkFreeMemory(device, m_labelStagingMemory, nullptr);    m_labelStagingMemory = VK_NULL_HANDLE; }

    // Layer images + framebuffers
    for (auto& L : m_layerImages)
    {
        if (L.framebuffer) vkDestroyFramebuffer(device, L.framebuffer, nullptr);
        if (L.view)        vkDestroyImageView(device, L.view, nullptr);
        if (L.image)       vkDestroyImage(device, L.image, nullptr);
        if (L.memory)      vkFreeMemory(device, L.memory, nullptr);
    }
    m_layerImages.clear();

    if (m_noAttachmentFramebuffer) {
        vkDestroyFramebuffer(device, m_noAttachmentFramebuffer, nullptr);
        m_noAttachmentFramebuffer = VK_NULL_HANDLE;
    }

    // Head pointer image
    if (m_headPointerView)   { vkDestroyImageView(device, m_headPointerView, nullptr); m_headPointerView   = VK_NULL_HANDLE; }
    if (m_headPointerImage)  { vkDestroyImage(device, m_headPointerImage, nullptr);    m_headPointerImage  = VK_NULL_HANDLE; }
    if (m_headPointerMemory) { vkFreeMemory(device, m_headPointerMemory, nullptr);     m_headPointerMemory = VK_NULL_HANDLE; }

    // Node buffer + counter
    if (m_nodeBuffer)          { vkDestroyBuffer(device, m_nodeBuffer, nullptr);          m_nodeBuffer          = VK_NULL_HANDLE; }
    if (m_nodeBufferMemory)    { vkFreeMemory(device, m_nodeBufferMemory, nullptr);       m_nodeBufferMemory    = VK_NULL_HANDLE; }
    if (m_atomicCounterBuffer) { vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr); m_atomicCounterBuffer = VK_NULL_HANDLE; }
    if (m_atomicCounterMemory) { vkFreeMemory(device, m_atomicCounterMemory, nullptr);    m_atomicCounterMemory = VK_NULL_HANDLE; }
    m_nodeCapacity = 0;

    // Union-find parents + region labels
    if (m_parentBuffer)       { vkDestroyBuffer(device, m_parentBuffer, nullptr);      m_parentBuffer       = VK_NULL_HANDLE; }
    if (m_parentBufferMemory) { vkFreeMemory(device, m_parentBufferMemory, nullptr);   m_parentBufferMemory = VK_NULL_HANDLE; }
    if (m_regionLabelBuffer)  { vkDestroyBuffer(device, m_regionLabelBuffer, nullptr); m_regionLabelBuffer  = VK_NULL_HANDLE; }
    if (m_regionLabelMemory)  { vkFreeMemory(device, m_regionLabelMemory, nullptr);    m_regionLabelMemory  = VK_NULL_HANDLE; }
    m_labelBufferSize = 0;
}

bool Sam2Renderer::createRenderPasses(VkDevice device)
{
    // ---- No-attachment render pass (collect + sort) ----
    {
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.subpassCount = 1;
        rp.pSubpasses   = &sub;
        if (vkCreateRenderPass(device, &rp, nullptr, &m_noAttachmentRenderPass) != VK_SUCCESS) return false;
    }

    // ---- Layer render pass (resolve output) ----
    {
        VkAttachmentDescription att{};
        att.format         = VK_FORMAT_R8G8B8A8_UNORM;
        att.samples        = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments    = &ref;

        std::array<VkSubpassDependency, 2> deps{};

        // Incoming: previous sampling (ImGui / resolve reads) or transfer read (readback)
        // must finish before we overwrite the attachment. PPLL SSBO visibility is handled
        // by the explicit memory barrier recorded before the render pass.
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = 0; // WAR: execution dependency is enough
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // Outgoing: color writes must be visible to later sampling (ImGui) and to the copy.
        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 1;
        rp.pAttachments    = &att;
        rp.subpassCount    = 1;
        rp.pSubpasses      = &sub;
        rp.dependencyCount = static_cast<uint32_t>(deps.size());
        rp.pDependencies   = deps.data();
        if (vkCreateRenderPass(device, &rp, nullptr, &m_layerRenderPass) != VK_SUCCESS) return false;
    }
    return true;
}

void Sam2Renderer::recordClearPhase(VkCommandBuffer cmd)
{
    // We clear the whole image anyway, so old contents don't matter:
    // oldLayout = UNDEFINED is valid on the first frame, after resize, and every frame after.
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask       = 0;
    ib.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image               = m_headPointerImage;
    ib.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // src = FRAGMENT_SHADER: waits for the previous frame's resolve reads (WAR).
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &ib);

    VkClearColorValue headClear{};
    headClear.uint32[0] = UFRenderer::UF_NULL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_headPointerImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &headClear, 1, &range);

    vkCmdFillBuffer(cmd, m_atomicCounterBuffer, 0, sizeof(uint32_t), 0);

    // TRANSFER_DST -> GENERAL for imageAtomicExchange / imageLoad / imageStore
    ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    ib.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // covers the counter fill
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 1, &ib);
}

std::vector<Sam2Renderer::CapturedLayer>
Sam2Renderer::captureSnapshot(VulkanContext& ctx,
                              VkBuffer vertexBuffer,
                              VkBuffer indexBuffer,
                              const std::vector<SceneObject>& objects,
                              const glm::mat4& viewProj,
                              const glm::mat4& view,
                              const glm::mat4& projection)
{
    m_projX = projection[0][0];
    m_projY = projection[1][1];

    std::vector<CapturedLayer> results(m_cfg.maxLayers);
    for (auto& r : results) { r.width = m_width; r.height = m_height; }

    auto submitAndWait = [&](auto&& recordFn, const char* tag)
    {
        vkResetCommandBuffer(m_captureCmd, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_captureCmd, &bi) != VK_SUCCESS)
            throw std::runtime_error(std::string("[Sam2] begin failed: ") + tag);

        recordFn(m_captureCmd);

        if (vkEndCommandBuffer(m_captureCmd) != VK_SUCCESS)
            throw std::runtime_error(std::string("[Sam2] end failed: ") + tag);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &m_captureCmd;

        vkResetFences(ctx.device, 1, &m_captureFence);
        if (vkQueueSubmit(ctx.graphics_queue, 1, &si, m_captureFence) != VK_SUCCESS)
            throw std::runtime_error(std::string("[Sam2] submit failed: ") + tag);
        vkWaitForFences(ctx.device, 1, &m_captureFence, VK_TRUE, UINT64_MAX);
    };

    // ---- Submit A: clear + collect + sort + union-find ----
    submitAndWait([&](VkCommandBuffer cmd)
    {
        recordClearPhase(cmd);

        recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj, view);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        recordSortPhase(cmd);
        recordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        recordUnionFindPhase(cmd);
    }, "A");

    // ---- Submit B_i: resolve layer i + region labels of layer i, copy both to staging ----
    for (uint32_t i = 0; i < m_cfg.maxLayers; ++i)
    {
        submitAndWait([&](VkCommandBuffer cmd)
        {
            // A fence does NOT make earlier GPU writes visible to later GPU commands.
            // This barrier makes the sort pass's SSBO/image writes visible to the resolve shader.
            recordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

            recordResolvePhase(cmd, i);
            recordCopyLayerToStaging(cmd, i);

            recordFullscreenPass(cmd, m_regionLabelsPipeline, i);
            recordCopyRegionLabelsToStaging(cmd);
        }, "B");

        results[i].rgba.resize(static_cast<size_t>(m_stagingBufferSize));
        std::memcpy(results[i].rgba.data(), m_stagingMapped, static_cast<size_t>(m_stagingBufferSize));

        results[i].regionIds.resize(size_t(m_width) * m_height);
        std::memcpy(results[i].regionIds.data(), m_labelStagingMapped, static_cast<size_t>(m_labelBufferSize));
    }

    std::cout << "[Sam2] Captured " << m_cfg.maxLayers
              << " layers @ " << m_width << "x" << m_height << std::endl;
    return results;
}

// =========================================================================
// Public: blit preview (layer 0) into the swapchain image
// =========================================================================

void Sam2Renderer::recordBlitPreviewToSwapchain(VkCommandBuffer cmd,
                                                VkImage swapchainImage,
                                                VkExtent2D swapchainExtent)
{
    if (m_layerImages.empty() || swapchainImage == VK_NULL_HANDLE) return;

    VkImage src = m_layerImages[0].image;
    const VkImageSubresourceRange colorRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // ---------------------------------------------------------------------
    // 1) Pre-blit transitions
    // ---------------------------------------------------------------------
    std::array<VkImageMemoryBarrier, 2> pre{};

    // Layer 0: SHADER_READ_ONLY_OPTIMAL (resolve pass finalLayout) -> TRANSFER_SRC_OPTIMAL
    pre[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre[0].srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    pre[0].oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pre[0].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[0].image               = src;
    pre[0].subresourceRange    = colorRange;

    // Swapchain image: UNDEFINED -> TRANSFER_DST_OPTIMAL (fully overwritten, old content irrelevant)
    pre[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre[1].srcAccessMask       = 0;
    pre[1].dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre[1].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    pre[1].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[1].image               = swapchainImage;
    pre[1].subresourceRange    = colorRange;

    // srcStage = COLOR_ATTACHMENT_OUTPUT does two jobs:
    //  - waits for the resolve pass's color writes into layer 0
    //  - chains with the acquire semaphore (waitDstStage = COLOR_ATTACHMENT_OUTPUT),
    //    so the swapchain layout transition happens only after the image is really available
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         static_cast<uint32_t>(pre.size()), pre.data());

    // ---------------------------------------------------------------------
    // 2) Blit
    //    vkCmdBlitImage converts formats: RGBA8_UNORM -> BGRA8_SRGB (swizzle + sRGB encode),
    //    and scales if the sizes differ.
    // ---------------------------------------------------------------------
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0]  = {0, 0, 0};
    blit.srcOffsets[1]  = {static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0]  = {0, 0, 0};
    blit.dstOffsets[1]  = {static_cast<int32_t>(swapchainExtent.width),
                           static_cast<int32_t>(swapchainExtent.height), 1};

    const bool sameSize = (m_width == swapchainExtent.width && m_height == swapchainExtent.height);

    vkCmdBlitImage(cmd,
                   src,            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit,
                   sameSize ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

    // ---------------------------------------------------------------------
    // 3) Post-blit transitions
    // ---------------------------------------------------------------------
    std::array<VkImageMemoryBarrier, 2> post{};

    // Layer 0: TRANSFER_SRC -> SHADER_READ_ONLY_OPTIMAL
    // (keeps previewImageLayout() true and matches what recordCopyLayerToStaging expects)
    post[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post[0].srcAccessMask       = 0;  // read-only access before, nothing to make visible
    post[0].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    post[0].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[0].newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[0].image               = src;
    post[0].subresourceRange    = colorRange;

    // Swapchain: TRANSFER_DST -> PRESENT_SRC_KHR
    // ImGui's render pass has initialLayout = PRESENT_SRC_KHR and LOAD_OP_LOAD,
    // so it will draw on top of our preview.
    post[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post[1].srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[1].dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    post[1].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[1].newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    post[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[1].image               = swapchainImage;
    post[1].subresourceRange    = colorRange;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                       | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         static_cast<uint32_t>(post.size()), post.data());
}