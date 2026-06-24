#include "depth_peeling_renderer.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "push_constants.h"
#include "mesh.h"

#include <iostream>
#include <array>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Init / Destroy / Recreate
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::init(VulkanContext &context, SwapchainManager &swapchain)
{
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;
    m_depthFormat = swapchain.getDepthFormat();

    if (!createPeelLayers(context, m_width, m_height))
        return false;
    if (!createRenderPasses(context.device, m_colorFormat, m_depthFormat))
        return false;
    if (!createFramebuffers(context.device, m_width, m_height))
        return false;
    if (!createSampler(context.device))
        return false;
    if (!createDescriptorSetLayout(context.device))
        return false;
    if (!createDescriptorPool(context.device))
        return false;
    if (!createDescriptorSets(context.device))
        return false;
    if (!createPipelineLayouts(context.device))
        return false;
    if (!createPipelines(context.device, swapchain.getRenderPass()))
        return false;
    if (!createCompositeFramebuffers(context.device, swapchain))
        return false;

    std::cout << "[DepthPeeling] Initialized with " << NUM_PEEL_LAYERS << " layers at "
              << m_width << "x" << m_height << std::endl;
    return true;
}

void DepthPeelingRenderer::destroy(VkDevice device)
{
    vkDeviceWaitIdle(device);

    vkDestroyPipeline(device, m_initPipeline, nullptr);
    vkDestroyPipeline(device, m_peelPipeline, nullptr);
    vkDestroyPipeline(device, m_compositePipeline, nullptr);

    vkDestroyPipelineLayout(device, m_scenePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, m_compositePipelineLayout, nullptr);

    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_peelDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_compositeDescriptorSetLayout, nullptr);

    vkDestroySampler(device, m_depthSampler, nullptr);
    vkDestroySampler(device, m_colorSampler, nullptr);

    for (auto fb : m_compositeFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }

    destroyPeelResources(device);

    vkDestroyRenderPass(device, m_peelRenderPass, nullptr);
    vkDestroyRenderPass(device, m_compositeRenderPass, nullptr);
}

void DepthPeelingRenderer::recreate(VulkanContext &context, SwapchainManager &swapchain)
{
    vkDeviceWaitIdle(context.device);

    // Destroy old layer resources and framebuffers
    for (auto fb : m_compositeFramebuffers)
    {
        vkDestroyFramebuffer(context.device, fb, nullptr);
    }
    m_compositeFramebuffers.clear();
    destroyPeelResources(context.device);

    // Recreate at new size
    m_width = swapchain.getExtent().width;
    m_height = swapchain.getExtent().height;

    createPeelLayers(context, m_width, m_height);
    createFramebuffers(context.device, m_width, m_height);
    createCompositeFramebuffers(context.device, swapchain);

    // Descriptor sets need updating with new image views
    createDescriptorSets(context.device);
}

// ─────────────────────────────────────────────────────────────────────────────
// Create Offscreen Peel Layers
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createPeelLayers(VulkanContext &context, uint32_t width, uint32_t height)
{
    m_layers.resize(NUM_PEEL_LAYERS);

    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++)
    {
        // Color: will be rendered to AND sampled in composite
        if (!createImage(context, width, height,
                         m_colorFormat,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         m_layers[i].colorImage, m_layers[i].colorMemory, m_layers[i].colorView))
        {
            std::cerr << "[DepthPeeling] Failed to create color image for layer " << i << std::endl;
            return false;
        }

        // Depth: will be rendered to AND sampled in next peel pass
        if (!createImage(context, width, height,
                         m_depthFormat,
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_DEPTH_BIT,
                         m_layers[i].depthImage, m_layers[i].depthMemory, m_layers[i].depthView))
        {
            std::cerr << "[DepthPeeling] Failed to create depth image for layer " << i << std::endl;
            return false;
        }
    }

    std::cout << "[DepthPeeling] Created " << NUM_PEEL_LAYERS << " offscreen layers." << std::endl;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render Passes
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createRenderPasses(VkDevice device, VkFormat colorFormat, VkFormat depthFormat)
{
    // ═══════════════════════════════════════════════════════════
    // Peel Render Pass: color + depth attachments (offscreen)
    // Used for BOTH init pass (layer 0) and peel passes (layer 1..N-1)
    // ═══════════════════════════════════════════════════════════
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = colorFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        // Dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies{};

        // Before: ensure previous reads are done
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // After: ensure writes are visible for next shader read
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments = attachments.data();
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        rpInfo.pDependencies = dependencies.data();

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_peelRenderPass) != VK_SUCCESS)
        {
            std::cerr << "[DepthPeeling] Failed to create peel render pass" << std::endl;
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Composite Render Pass: renders onto swapchain image
    // ═══════════════════════════════════════════════════════════
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = VK_FORMAT_B8G8R8A8_SRGB; // Swapchain format
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &colorAttachment;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_compositeRenderPass) != VK_SUCCESS)
        {
            std::cerr << "[DepthPeeling] Failed to create composite render pass" << std::endl;
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Framebuffers (one per peel layer)
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createFramebuffers(VkDevice device, uint32_t width, uint32_t height)
{
    for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++)
    {
        std::array<VkImageView, 2> attachments = {
            m_layers[i].colorView,
            m_layers[i].depthView};

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_peelRenderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = width;
        fbInfo.height = height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_layers[i].framebuffer) != VK_SUCCESS)
        {
            std::cerr << "[DepthPeeling] Failed to create framebuffer for layer " << i << std::endl;
            return false;
        }
    }
    return true;
}

bool DepthPeelingRenderer::createCompositeFramebuffers(VkDevice device, SwapchainManager &swapchain)
{
    // Need to access swapchain image views — add a getter to SwapchainManager
    // For now, we'll use the swapchain's render pass and framebuffers directly
    // The composite will render into the swapchain's existing framebuffers
    // So we don't need separate composite framebuffers if we reuse SwapchainManager's render pass.

    // Actually, we need our own because our composite render pass is different (no depth attachment)
    // We'll create framebuffers wrapping swapchain image views with our composite render pass

    m_compositeFramebuffers.resize(swapchain.getImageCount());

    for (uint32_t i = 0; i < swapchain.getImageCount(); i++)
    {
        VkImageView attachment = swapchain.getImageView(i);

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_compositeRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &attachment;
        fbInfo.width = swapchain.getExtent().width;
        fbInfo.height = swapchain.getExtent().height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_compositeFramebuffers[i]) != VK_SUCCESS)
        {
            std::cerr << "[DepthPeeling] Failed to create composite framebuffer " << i << std::endl;
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampler
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createSampler(VkDevice device)
{
    // Depth sampler: nearest filtering, clamp to edge
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &m_depthSampler) != VK_SUCCESS)
    {
        return false;
    }

    // Color sampler: linear filtering for smooth composite
    VkSamplerCreateInfo colorSamplerInfo = samplerInfo;
    colorSamplerInfo.magFilter = VK_FILTER_LINEAR;
    colorSamplerInfo.minFilter = VK_FILTER_LINEAR;

    if (vkCreateSampler(device, &colorSamplerInfo, nullptr, &m_colorSampler) != VK_SUCCESS)
    {
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor Set Layouts
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createDescriptorSetLayout(VkDevice device)
{
    // ═══ Peel layout: binding 0 = previous layer depth texture ═══
    {
        VkDescriptorSetLayoutBinding depthBinding{};
        depthBinding.binding = 0;
        depthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        depthBinding.descriptorCount = 1;
        depthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &depthBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_peelDescriptorSetLayout) != VK_SUCCESS)
        {
            return false;
        }
    }

    // ═══ Composite layout: binding 0..N-1 = all color layer textures ═══
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(NUM_PEEL_LAYERS);
        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_compositeDescriptorSetLayout) != VK_SUCCESS)
        {
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor Pool & Sets
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createDescriptorPool(VkDevice device)
{
    // Peel sets: (NUM_PEEL_LAYERS - 1) sets, each with 1 combined image sampler
    // Composite set: 1 set with NUM_PEEL_LAYERS combined image samplers
    uint32_t totalSamplers = (NUM_PEEL_LAYERS - 1) + NUM_PEEL_LAYERS;
    uint32_t totalSets = (NUM_PEEL_LAYERS - 1) + 1 + 1; // Peel sets + composite set + dummy set

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = totalSamplers;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = totalSets;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
    {
        return false;
    }

    return true;
}

bool DepthPeelingRenderer::createDescriptorSets(VkDevice device)
{
    // ═══ Peel descriptor sets: layer i reads depth from layer i-1 ═══
    m_peelDescriptorSets.resize(NUM_PEEL_LAYERS - 1);

    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_peelDescriptorSetLayout;
        vkAllocateDescriptorSets(device, &allocInfo, &m_dummyDescriptorSet);

        // Point it at layer 0's depth (won't actually be sampled in init shader)
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = m_depthSampler;
        imageInfo.imageView = m_layers[0].depthView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_dummyDescriptorSet;
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    for (uint32_t i = 0; i < NUM_PEEL_LAYERS - 1; i++)
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_peelDescriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &m_peelDescriptorSets[i]) != VK_SUCCESS)
        {
            return false;
        }

        // Write: bind layer[i]'s depth as sampler (this is the PREVIOUS layer for peel pass i+1)
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = m_depthSampler;
        imageInfo.imageView = m_layers[i].depthView; // Previous layer's depth
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_peelDescriptorSets[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    // ═══ Composite descriptor set: all color layers ═══
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_compositeDescriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &m_compositeDescriptorSet) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkDescriptorImageInfo> imageInfos(NUM_PEEL_LAYERS);
        std::vector<VkWriteDescriptorSet> writes(NUM_PEEL_LAYERS);

        for (uint32_t i = 0; i < NUM_PEEL_LAYERS; i++)
        {
            imageInfos[i].sampler = m_colorSampler;
            imageInfos[i].imageView = m_layers[i].colorView;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_compositeDescriptorSet;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &imageInfos[i];
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Layouts
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createPipelineLayouts(VkDevice device)
{
    // ═══ Scene pipeline layout: push constants + optional prev depth set ═══
    {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(ScenePushConstants);

        // Set 0 = peel descriptor (prev depth) — only used in peel passes, but layout must match
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_peelDescriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_scenePipelineLayout) != VK_SUCCESS)
        {
            return false;
        }
    }

    // ═══ Composite pipeline layout: color layer samplers ═══
    {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t); // numLayers

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_compositeDescriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_compositePipelineLayout) != VK_SUCCESS)
        {
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipelines
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createPipelines(VkDevice device, VkRenderPass swapchainRenderPass)
{
    std::string shaderDir = "shaders";

    VkShaderModule vertModule = ShaderUtils::createShaderModule(device, shaderDir + "/geometry.vert.spv");
    VkShaderModule initFragModule = ShaderUtils::createShaderModule(device, shaderDir + "/depth_init.frag.spv");
    VkShaderModule peelFragModule = ShaderUtils::createShaderModule(device, shaderDir + "/depth_peel.frag.spv");
    VkShaderModule compositeVertModule = ShaderUtils::createShaderModule(device, shaderDir + "/composite.vert.spv");
    VkShaderModule compositeFragModule = ShaderUtils::createShaderModule(device, shaderDir + "/composite.frag.spv");

    if (!vertModule || !initFragModule || !peelFragModule || !compositeVertModule || !compositeFragModule)
    {
        std::cerr << "[DepthPeeling] Failed to load one or more shaders" << std::endl;
        return false;
    }

    VkVertexInputBindingDescription binding = Vertex::getBindingDescription();
    std::vector<VkVertexInputAttributeDescription> attributes = Vertex::getAttributeDescriptions();

    PipelineBuilder builder;

    // ═══ Init Pipeline (layer 0 — standard depth test) ═══
    m_initPipeline = builder
                         .setVertexShader(vertModule)
                         .setFragmentShader(initFragModule)
                         .setVertexInput({binding}, attributes)
                         .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                         .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                         .enableDepthTest(true, VK_COMPARE_OP_LESS)
                         .disableBlending()
                         .setPipelineLayout(m_scenePipelineLayout)
                         .setRenderPass(m_peelRenderPass, 0)
                         .build(device);

    // ═══ Peel Pipeline (layer 1..N — discard ≤ previous depth) ═══
    builder.clear();
    m_peelPipeline = builder
                         .setVertexShader(vertModule)
                         .setFragmentShader(peelFragModule)
                         .setVertexInput({binding}, attributes)
                         .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                         .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                         .enableDepthTest(true, VK_COMPARE_OP_LESS)
                         .disableBlending()
                         .setPipelineLayout(m_scenePipelineLayout)
                         .setRenderPass(m_peelRenderPass, 0)
                         .build(device);

    // ═══ Composite Pipeline (fullscreen quad, blend layers) ═══
    builder.clear();
    m_compositePipeline = builder
                              .setVertexShader(compositeVertModule)
                              .setFragmentShader(compositeFragModule)
                              .setNoVertexInput()
                              .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
                              .setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                              .disableDepthTest()
                              .disableBlending()
                              .setPipelineLayout(m_compositePipelineLayout)
                              .setRenderPass(m_compositeRenderPass, 0)
                              .build(device);

    // Cleanup shader modules
    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, initFragModule, nullptr);
    vkDestroyShaderModule(device, peelFragModule, nullptr);
    vkDestroyShaderModule(device, compositeVertModule, nullptr);
    vkDestroyShaderModule(device, compositeFragModule, nullptr);

    if (!m_initPipeline || !m_peelPipeline || !m_compositePipeline)
    {
        std::cerr << "[DepthPeeling] Failed to create one or more pipelines" << std::endl;
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Record Commands — THE CORE MULTI-PASS LOGIC
// ─────────────────────────────────────────────────────────────────────────────

void DepthPeelingRenderer::recordCommands(VkCommandBuffer cmd,
                                           VulkanContext& context,
                                           SwapchainManager& swapchain,
                                           VkBuffer vertexBuffer,
                                           VkBuffer indexBuffer,
                                           const std::vector<SceneObject>& objects,
                                           VkPipelineLayout sceneLayout,
                                           const glm::mat4& viewProj,
                                           uint32_t swapchainImageIndex)
{
    VkExtent2D extent = swapchain.getExtent();

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(extent.height);
    viewport.width = static_cast<float>(extent.width);
    viewport.height = -static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}}; // Transparent black
    clearValues[1].depthStencil = {1.0f, 0};

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};

    auto drawAllObjects = [&](VkCommandBuffer cmd) {
        for (const auto& obj : objects) {
            ScenePushConstants pc{};
            pc.model = glm::mat4(1.0f);   // Could be per-object transform
            pc.viewProj = viewProj;
            pc.objectID = obj.objectID;

            vkCmdPushConstants(cmd, m_scenePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ScenePushConstants), &pc);

            vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexOffset, 0, 0);
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // PASS 0: Init — render first (nearest) layer
    // ═══════════════════════════════════════════════════════════════════
    {
        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = m_peelRenderPass;
        rpInfo.framebuffer = m_layers[0].framebuffer;
        rpInfo.renderArea = {.offset = {0, 0}, .extent = extent};
        rpInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_initPipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_scenePipelineLayout, 0, 1,
                        &m_dummyDescriptorSet, 0, nullptr);

        // Draw all objects
        // Quad 1
        drawAllObjects(cmd);
        vkCmdEndRenderPass(cmd);
    }

    // ═══════════════════════════════════════════════════════════════════
    // PASS 1..N-1: Peel — render subsequent layers
    // ═══════════════════════════════════════════════════════════════════
    for (uint32_t layer = 1; layer < NUM_PEEL_LAYERS; layer++)
    {
        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = m_peelRenderPass;
        rpInfo.framebuffer = m_layers[layer].framebuffer;
        rpInfo.renderArea = {.offset = {0, 0}, .extent = extent};
        rpInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_peelPipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Bind previous layer's depth as descriptor set 0
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_scenePipelineLayout, 0, 1,
                                &m_peelDescriptorSets[layer - 1], 0, nullptr);

        drawAllObjects(cmd);
        vkCmdEndRenderPass(cmd);
    }

    // ═══════════════════════════════════════════════════════════════════
    // COMPOSITE PASS: Blend all layers back-to-front
    // ═══════════════════════════════════════════════════════════════════
    {
        VkClearValue compositeClear{};
        compositeClear.color = {{0.02f, 0.02f, 0.08f, 1.0f}}; // Background

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = m_compositeRenderPass;
        rpInfo.framebuffer = m_compositeFramebuffers[swapchainImageIndex];
        rpInfo.renderArea = {.offset = {0, 0}, .extent = extent};
        rpInfo.clearValueCount = 1;
        rpInfo.pClearValues = &compositeClear;

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Bind all color layers
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_compositePipelineLayout, 0, 1,
                                &m_compositeDescriptorSet, 0, nullptr);

        // Push number of layers
        uint32_t numLayers = NUM_PEEL_LAYERS;
        vkCmdPushConstants(cmd, m_compositePipelineLayout,
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(uint32_t), &numLayers);

        // Draw fullscreen triangle (3 vertices, no VBO needed)
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Creation Helper
// ─────────────────────────────────────────────────────────────────────────────

bool DepthPeelingRenderer::createImage(VulkanContext &context, uint32_t width, uint32_t height,
                                       VkFormat format, VkImageUsageFlags usage,
                                       VkImageAspectFlags aspect,
                                       VkImage &image, VkDeviceMemory &memory, VkImageView &view)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(context.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(context.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
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

    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
    {
        return false;
    }

    return true;
}

uint32_t DepthPeelingRenderer::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                              VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return 0;
}

void DepthPeelingRenderer::destroyPeelResources(VkDevice device)
{
    for (auto &layer : m_layers)
    {
        if (layer.framebuffer)
            vkDestroyFramebuffer(device, layer.framebuffer, nullptr);
        if (layer.colorView)
            vkDestroyImageView(device, layer.colorView, nullptr);
        if (layer.colorImage)
            vkDestroyImage(device, layer.colorImage, nullptr);
        if (layer.colorMemory)
            vkFreeMemory(device, layer.colorMemory, nullptr);
        if (layer.depthView)
            vkDestroyImageView(device, layer.depthView, nullptr);
        if (layer.depthImage)
            vkDestroyImage(device, layer.depthImage, nullptr);
        if (layer.depthMemory)
            vkFreeMemory(device, layer.depthMemory, nullptr);
    }
    m_layers.clear();
}