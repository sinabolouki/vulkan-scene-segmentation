#include "ppll_renderer.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "mesh.h"
#include "push_constants.h"

#include <iostream>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// Init / Destroy / Recreate
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::init(VulkanContext& context, SwapchainManager& swapchain) {
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    if (!createHeadPointerImage(context, m_width, m_height)) return false;
    if (!createFragmentNodeBuffer(context, m_width, m_height)) return false;
    if (!createAtomicCounterBuffer(context)) return false;
    if (!createRenderPasses(context.device, swapchain.getImageFormat())) return false;
    if (!createDescriptorSetLayouts(context.device)) return false;
    if (!createDescriptorPool(context.device)) return false;
    if (!createDescriptorSets(context.device)) return false;
    if (!createPipelineLayouts(context.device)) return false;
    if (!createPipelines(context.device)) return false;
    if (!createFramebuffers(context.device, swapchain)) return false;

    std::cout << "[PPLL] Initialized:" << std::endl;
    std::cout << "[PPLL]   Resolution:    " << m_width << "x" << m_height << std::endl;
    std::cout << "[PPLL]   Node capacity: " << m_nodeCapacity << " ("
              << (m_nodeCapacity * sizeof(FragmentNode)) / (1024 * 1024) << " MB)" << std::endl;

    return true;
}

void PPLLRenderer::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_collectPipeline, nullptr);
    vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    vkDestroyPipelineLayout(device, m_collectPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, m_resolvePipelineLayout, nullptr);

    vkDestroyFramebuffer(device, m_collectFramebuffer, nullptr);
    for (auto fb : m_resolveFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);

    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_ppllDescLayout, nullptr);

    vkDestroyRenderPass(device, m_collectRenderPass, nullptr);
    vkDestroyRenderPass(device, m_resolveRenderPass, nullptr);

    destroyDataStructures(device);
}

void PPLLRenderer::recreate(VulkanContext& context, SwapchainManager& swapchain) {
    vkDeviceWaitIdle(context.device);

    vkDestroyFramebuffer(context.device, m_collectFramebuffer, nullptr);
    m_collectFramebuffer = VK_NULL_HANDLE;
    for (auto fb : m_resolveFramebuffers)
        vkDestroyFramebuffer(context.device, fb, nullptr);
    m_resolveFramebuffers.clear();

    // Destroy old descriptor pool (sets are freed with it)
    vkDestroyDescriptorPool(context.device, m_descriptorPool, nullptr);
    m_descriptorPool = VK_NULL_HANDLE;

    destroyDataStructures(context.device);

    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    createHeadPointerImage(context, m_width, m_height);
    createFragmentNodeBuffer(context, m_width, m_height);
    createAtomicCounterBuffer(context);
    createDescriptorPool(context.device);
    createDescriptorSets(context.device);
    createFramebuffers(context.device, swapchain);
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Structure Creation (same as before)
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createHeadPointerImage(VulkanContext& context, uint32_t width, uint32_t height) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R32_UINT;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(context.device, &imageInfo, nullptr, &m_headPointerImage) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, m_headPointerImage, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &m_headPointerMemory) != VK_SUCCESS) return false;
    vkBindImageMemory(context.device, m_headPointerImage, m_headPointerMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_headPointerImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32_UINT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &m_headPointerView) != VK_SUCCESS) return false;
    return true;
}

bool PPLLRenderer::createFragmentNodeBuffer(VulkanContext& context, uint32_t width, uint32_t height) {
    m_nodeCapacity = width * height * AVG_FRAGS_PER_PIXEL;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(m_nodeCapacity) * sizeof(FragmentNode);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufferInfo, nullptr, &m_nodeBuffer) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, m_nodeBuffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &m_nodeBufferMemory) != VK_SUCCESS) return false;
    vkBindBufferMemory(context.device, m_nodeBuffer, m_nodeBufferMemory, 0);
    return true;
}

bool PPLLRenderer::createAtomicCounterBuffer(VulkanContext& context) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(uint32_t);
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufferInfo, nullptr, &m_atomicCounterBuffer) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, m_atomicCounterBuffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &m_atomicCounterMemory) != VK_SUCCESS) return false;
    vkBindBufferMemory(context.device, m_atomicCounterBuffer, m_atomicCounterMemory, 0);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render Passes
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createRenderPasses(VkDevice device, VkFormat swapchainFormat) {
    // ═══ Collect Render Pass: NO attachments ═══
    // Fragment shader only writes to SSBO/storage image via imageStore + atomics.
    // We still need a render pass for rasterization to generate fragments.
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = nullptr;
        subpass.pDepthStencilAttachment = nullptr;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 0;
        rpInfo.pAttachments = nullptr;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 0;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_collectRenderPass) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create collect render pass" << std::endl;
            return false;
        }
    }

    // ═══ Resolve Render Pass: one color attachment (swapchain) ═══
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
        dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAtt;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dep;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_resolveRenderPass) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create resolve render pass" << std::endl;
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Framebuffers
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    // ═══ Collect framebuffer: no attachments, just dimensions ═══
    {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_collectRenderPass;
        fbInfo.attachmentCount = 0;
        fbInfo.pAttachments = nullptr;
        fbInfo.width = m_width;
        fbInfo.height = m_height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_collectFramebuffer) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create collect framebuffer" << std::endl;
            return false;
        }
    }

    // ═══ Resolve framebuffers: one per swapchain image ═══
    m_resolveFramebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        VkImageView view = swapchain.getImageView(i);

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_resolveRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &view;
        fbInfo.width = swapchain.getExtent().width;
        fbInfo.height = swapchain.getExtent().height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_resolveFramebuffers[i]) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create resolve framebuffer " << i << std::endl;
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor Set Layouts
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createDescriptorSetLayouts(VkDevice device) {
    // Single layout shared by collect and resolve:
    //   binding 0: head pointer image (storage image, r32ui)
    //   binding 1: fragment node buffer (storage buffer)
    //   binding 2: atomic counter buffer (storage buffer)

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    // Binding 0: Head pointer image
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 1: Node buffer
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 2: Atomic counter
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_ppllDescLayout) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor Pool & Sets
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2; // node buffer + atomic counter

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1; // Single descriptor set shared by both passes

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to create descriptor pool" << std::endl;
        return false;
    }

    return true;
}

bool PPLLRenderer::createDescriptorSets(VkDevice device) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_ppllDescLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &m_ppllDescSet) != VK_SUCCESS) {
        std::cerr << "[PPLL] Failed to allocate descriptor set" << std::endl;
        return false;
    }

    // ═══ Write descriptors ═══
    // Binding 0: Head pointer storage image
    VkDescriptorImageInfo headImageInfo{};
    headImageInfo.imageView = m_headPointerView;
    headImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // sampler not needed for storage images

    // Binding 1: Node buffer
    VkDescriptorBufferInfo nodeBufferInfo{};
    nodeBufferInfo.buffer = m_nodeBuffer;
    nodeBufferInfo.offset = 0;
    nodeBufferInfo.range = VK_WHOLE_SIZE;

    // Binding 2: Atomic counter
    VkDescriptorBufferInfo counterBufferInfo{};
    counterBufferInfo.buffer = m_atomicCounterBuffer;
    counterBufferInfo.offset = 0;
    counterBufferInfo.range = sizeof(uint32_t);

    std::array<VkWriteDescriptorSet, 3> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_ppllDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &headImageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_ppllDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &nodeBufferInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = m_ppllDescSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &counterBufferInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Layouts
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createPipelineLayouts(VkDevice device) {
    // ═══ Collect pipeline layout: push constants + PPLL descriptor set ═══
    {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(ScenePushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_ppllDescLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_collectPipelineLayout) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create collect pipeline layout" << std::endl;
            return false;
        }
    }

    // ═══ Resolve pipeline layout: PPLL descriptor set + screen size push constant ═══
    {
        // Push constant for resolve: just screen dimensions
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t) * 2; // width, height

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_ppllDescLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_resolvePipelineLayout) != VK_SUCCESS) {
            std::cerr << "[PPLL] Failed to create resolve pipeline layout" << std::endl;
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipelines
// ─────────────────────────────────────────────────────────────────────────────

bool PPLLRenderer::createPipelines(VkDevice device) {
    std::string shaderDir = "shaders";

    VkShaderModule collectVert = ShaderUtils::createShaderModule(device, shaderDir + "/ppll_collect.vert.spv");
    VkShaderModule collectFrag = ShaderUtils::createShaderModule(device, shaderDir + "/ppll_collect.frag.spv");
    VkShaderModule resolveVert = ShaderUtils::createShaderModule(device, shaderDir + "/ppll_resolve.vert.spv");
    VkShaderModule resolveFrag = ShaderUtils::createShaderModule(device, shaderDir + "/ppll_resolve.frag.spv");

    if (!collectVert || !collectFrag || !resolveVert || !resolveFrag) {
        std::cerr << "[PPLL] Failed to load shaders" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription binding = Vertex::getBindingDescription();
    std::vector<VkVertexInputAttributeDescription> attributes = Vertex::getAttributeDescriptions();

    PipelineBuilder builder;

    // ═══ Collect Pipeline ═══
    // - Rasterizes geometry to generate fragments
    // - NO depth test (we want ALL fragments)
    // - NO color attachment (output goes to SSBO)
    // - Fragment shader inserts into linked list
    m_collectPipeline = builder
        .setVertexShader(collectVert)
        .setFragmentShader(collectFrag)
        .setVertexInput({binding}, attributes)
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE) // No culling — capture all
        .disableDepthTest()
        .setNoColorAttachments()
        .setPipelineLayout(m_collectPipelineLayout)
        .setRenderPass(m_collectRenderPass, 0)
        .build(device);

    if (!m_collectPipeline) {
        std::cerr << "[PPLL] Failed to create collect pipeline" << std::endl;
        return false;
    }

    // ═══ Resolve Pipeline ═══
    // - Fullscreen triangle (no vertex input)
    // - Reads linked list, sorts, composites
    // - Outputs to swapchain color attachment
    builder.clear();
    m_resolvePipeline = builder
        .setVertexShader(resolveVert)
        .setFragmentShader(resolveFrag)
        .setNoVertexInput()
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .disableDepthTest()
        .disableBlending()
        .setPipelineLayout(m_resolvePipelineLayout)
        .setRenderPass(m_resolveRenderPass, 0)
        .build(device);

    if (!m_resolvePipeline) {
        std::cerr << "[PPLL] Failed to create resolve pipeline" << std::endl;
        return false;
    }

    // Cleanup shader modules
    vkDestroyShaderModule(device, collectVert, nullptr);
    vkDestroyShaderModule(device, collectFrag, nullptr);
    vkDestroyShaderModule(device, resolveVert, nullptr);
    vkDestroyShaderModule(device, resolveFrag, nullptr);

    std::cout << "[PPLL] Pipelines created." << std::endl;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Command Recording
// ─────────────────────────────────────────────────────────────────────────────

void PPLLRenderer::recordCommands(VkCommandBuffer cmd,
                                   VulkanContext& context,
                                   SwapchainManager& swapchain,
                                   VkBuffer vertexBuffer,
                                   VkBuffer indexBuffer,
                                   const std::vector<SceneObject>& objects,
                                   const glm::mat4& viewProj,
                                   uint32_t swapchainImageIndex) {
    VkExtent2D extent = swapchain.getExtent();

    // Phase 1: Clear
    recordClearPhase(cmd);

    // Phase 2: Collect
    recordCollectPhase(cmd, vertexBuffer, indexBuffer, objects, viewProj, extent);

    // Phase 3: Barrier
    recordBarrier(cmd);

    // Phase 4: Resolve
    recordResolvePhase(cmd, swapchainImageIndex, extent);
}

void PPLLRenderer::recordClearPhase(VkCommandBuffer cmd) {
    // Transition head image for clear
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

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    // Clear head image to null
    VkClearColorValue clearValue{};
    clearValue.uint32[0] = PPLL_NULL;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_headPointerImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);

    // Reset atomic counter
    vkCmdFillBuffer(cmd, m_atomicCounterBuffer, 0, sizeof(uint32_t), 0);

    // Transition head image back to GENERAL for shader access
    imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &memBarrier, 0, nullptr, 1, &imgBarrier);
}

void PPLLRenderer::recordCollectPhase(VkCommandBuffer cmd,
                                       VkBuffer vertexBuffer,
                                       VkBuffer indexBuffer,
                                       const std::vector<SceneObject>& objects,
                                       const glm::mat4& viewProj,
                                       VkExtent2D extent) {
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_collectRenderPass;
    rpInfo.framebuffer = m_collectFramebuffer;
    rpInfo.renderArea = {{0, 0}, extent};
    rpInfo.clearValueCount = 0;
    rpInfo.pClearValues = nullptr;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_collectPipeline);

    // Dynamic viewport/scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(extent.height);
    viewport.width = static_cast<float>(extent.width);
    viewport.height = -static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind PPLL descriptor set
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_collectPipelineLayout, 0, 1,
                            &m_ppllDescSet, 0, nullptr);

    // Bind geometry
    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // Draw all objects
    for (const auto& obj : objects) {
        ScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = viewProj;
        pc.objectID = obj.objectID;

        vkCmdPushConstants(cmd, m_collectPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(ScenePushConstants), &pc);

        vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
}

void PPLLRenderer::recordBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &memBarrier, 0, nullptr, 0, nullptr);
}

void PPLLRenderer::recordResolvePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex, VkExtent2D extent) {
    VkClearValue clearValue{};
    clearValue.color = {{0.02f, 0.02f, 0.08f, 1.0f}};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_resolveRenderPass;
    rpInfo.framebuffer = m_resolveFramebuffers[swapchainImageIndex];
    rpInfo.renderArea = {{0, 0}, extent};
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clearValue;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_resolvePipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind PPLL descriptor set (same data, read-only now)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_resolvePipelineLayout, 0, 1,
                            &m_ppllDescSet, 0, nullptr);

    // Push screen dimensions
    uint32_t screenSize[2] = {extent.width, extent.height};
    vkCmdPushConstants(cmd, m_resolvePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(screenSize), screenSize);

    // Fullscreen triangle (3 vertices, no VBO)
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

uint32_t PPLLRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return 0;
}

void PPLLRenderer::destroyDataStructures(VkDevice device) {
    if (m_headPointerView) { vkDestroyImageView(device, m_headPointerView, nullptr); m_headPointerView = VK_NULL_HANDLE; }
    if (m_headPointerImage) { vkDestroyImage(device, m_headPointerImage, nullptr); m_headPointerImage = VK_NULL_HANDLE; }
    if (m_headPointerMemory) { vkFreeMemory(device, m_headPointerMemory, nullptr); m_headPointerMemory = VK_NULL_HANDLE; }
    if (m_nodeBuffer) { vkDestroyBuffer(device, m_nodeBuffer, nullptr); m_nodeBuffer = VK_NULL_HANDLE; }
    if (m_nodeBufferMemory) { vkFreeMemory(device, m_nodeBufferMemory, nullptr); m_nodeBufferMemory = VK_NULL_HANDLE; }
    if (m_atomicCounterBuffer) { vkDestroyBuffer(device, m_atomicCounterBuffer, nullptr); m_atomicCounterBuffer = VK_NULL_HANDLE; }
    if (m_atomicCounterMemory) { vkFreeMemory(device, m_atomicCounterMemory, nullptr); m_atomicCounterMemory = VK_NULL_HANDLE; }
    m_nodeCapacity = 0;
}