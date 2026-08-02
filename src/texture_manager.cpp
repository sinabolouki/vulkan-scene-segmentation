#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "texture_manager.h"
#include "vulkan_context.h"
#include <iostream>
#include <cstring>

bool TextureManager::init(VulkanContext& context) {
    if (!createSampler(context.device)) return false;
    if (!createPlaceholderTexture(context)) return false;
    if (!createDescriptorSetLayout(context.device)) return false;
    if (!createDescriptorPool(context.device)) return false;
    if (!createDescriptorSet(context.device)) return false;
    updateDescriptorSet(context.device);
    return true;
}

void TextureManager::destroy(VkDevice device) {
    for (auto& tex : m_textures) {
        if (tex.view) vkDestroyImageView(device, tex.view, nullptr);
        if (tex.image) vkDestroyImage(device, tex.image, nullptr);
        if (tex.memory) vkFreeMemory(device, tex.memory, nullptr);
    }
    m_textures.clear();

    if (m_placeholder.view) vkDestroyImageView(device, m_placeholder.view, nullptr);
    if (m_placeholder.image) vkDestroyImage(device, m_placeholder.image, nullptr);
    if (m_placeholder.memory) vkFreeMemory(device, m_placeholder.memory, nullptr);

    if (m_sampler) vkDestroySampler(device, m_sampler, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    if (m_descriptorSetLayout) vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
}

bool TextureManager::loadTextures(VulkanContext& context, const std::vector<std::string>& filePaths) {
    m_loadedPaths.clear();

    for (const auto& path : filePaths) {
        int width, height, channels;
        stbi_set_flip_vertically_on_load(false);
        uint8_t* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4); // Force RGBA

        if (!pixels) {
            std::cerr << "[Textures] Failed to load: " << path
                      << " (" << stbi_failure_reason() << ")" << std::endl;
            // Create a magenta placeholder so indices stay consistent
            uint8_t magenta[4] = {255, 0, 255, 255};
            Texture tex;
            createTextureFromPixels(context, magenta, 1, 1, 4, tex);
            m_textures.push_back(tex);
            m_loadedPaths.push_back(path);
            continue;
        }

        Texture tex;
        if (!createTextureFromPixels(context, pixels, (uint32_t)width, (uint32_t)height, 4, tex)) {
            std::cerr << "[Textures] Failed to create Vulkan image for: " << path << std::endl;
            stbi_image_free(pixels);
            return false;
        }

        stbi_image_free(pixels);
        m_textures.push_back(tex);
        m_loadedPaths.push_back(path);

        std::cout << "[Textures] Loaded: " << path
                  << " (" << width << "x" << height << ")" << std::endl;
    }

    // Update descriptor set with all loaded textures
    updateDescriptorSet(context.device);

    std::cout << "[Textures] Total loaded: " << m_textures.size() << std::endl;
    return true;
}

bool TextureManager::createTextureFromPixels(VulkanContext& context,
                                              const uint8_t* pixels,
                                              uint32_t width, uint32_t height,
                                              uint32_t channels, Texture& outTexture) {
    VkDeviceSize imageSize = width * height * 4; // Always RGBA

    // ─── Staging buffer ──────────────────────────────────────────────────
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = imageSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(context.device, &bufInfo, nullptr, &stagingBuffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(context.device, stagingBuffer, &memReq);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(context.device, &allocInfo, nullptr, &stagingMemory);
    vkBindBufferMemory(context.device, stagingBuffer, stagingMemory, 0);

    void* data;
    vkMapMemory(context.device, stagingMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, imageSize);
    vkUnmapMemory(context.device, stagingMemory);

    // ─── Create image ────────────────────────────────────────────────────
    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imgInfo.extent = {width, height, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(context.device, &imgInfo, nullptr, &outTexture.image) != VK_SUCCESS) return false;

    vkGetImageMemoryRequirements(context.device, outTexture.image, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(context.physical_device, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &outTexture.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(context.device, outTexture.image, outTexture.memory, 0);

    // ─── Copy staging → image via one-shot command buffer ────────────────
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAlloc.commandPool = context.command_pool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(context.device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition to TRANSFER_DST
    transitionImageLayout(cmd, outTexture.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, outTexture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ_ONLY
    transitionImageLayout(cmd, outTexture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphics_queue);

    vkFreeCommandBuffers(context.device, context.command_pool, 1, &cmd);
    vkDestroyBuffer(context.device, stagingBuffer, nullptr);
    vkFreeMemory(context.device, stagingMemory, nullptr);

    // ─── Create image view ───────────────────────────────────────────────
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = outTexture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(context.device, &viewInfo, nullptr, &outTexture.view) != VK_SUCCESS) return false;

    return true;
}

bool TextureManager::createPlaceholderTexture(VulkanContext& context) {
    uint8_t white[4] = {255, 255, 255, 255};
    return createTextureFromPixels(context, white, 1, 1, 4, m_placeholder);
}

void TextureManager::transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                            VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkPipelineStageFlags srcStage, dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool TextureManager::createSampler(VkDevice device) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = 16.0f;
    info.maxLod = 1.0f;
    return vkCreateSampler(device, &info, nullptr, &m_sampler) == VK_SUCCESS;
}

bool TextureManager::createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = MAX_TEXTURES;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = 1;
    info.pBindings = &binding;
    return vkCreateDescriptorSetLayout(device, &info, nullptr, &m_descriptorSetLayout) == VK_SUCCESS;
}

bool TextureManager::createDescriptorPool(VkDevice device) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = MAX_TEXTURES;

    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.poolSizeCount = 1;
    info.pPoolSizes = &poolSize;
    info.maxSets = 1;
    return vkCreateDescriptorPool(device, &info, nullptr, &m_descriptorPool) == VK_SUCCESS;
}

bool TextureManager::createDescriptorSet(VkDevice device) {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    return vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet) == VK_SUCCESS;
}

void TextureManager::updateDescriptorSet(VkDevice device) {
    std::vector<VkDescriptorImageInfo> imageInfos(MAX_TEXTURES);

    for (uint32_t i = 0; i < MAX_TEXTURES; i++) {
        if (i < m_textures.size()) {
            imageInfos[i].sampler = m_sampler;
            imageInfos[i].imageView = m_textures[i].view;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else {
            // Fill remaining slots with placeholder
            imageInfos[i].sampler = m_sampler;
            imageInfos[i].imageView = m_placeholder.view;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = m_descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = MAX_TEXTURES;
    write.pImageInfo = imageInfos.data();

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

uint32_t TextureManager::findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                         VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return 0;
}