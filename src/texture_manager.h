#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

struct VulkanContext;

class TextureManager {
public:
    static constexpr uint32_t MAX_TEXTURES = 64;

    bool init(VulkanContext& context);
    void destroy(VkDevice device);

    // Load all textures from the given file paths. Returns indices matching input order.
    bool loadTextures(VulkanContext& context, const std::vector<std::string>& filePaths);

    // Get the loaded texture paths (in order of their indices)
    const std::vector<std::string>& getLoadedPaths() const { return m_loadedPaths; }

    uint32_t getTextureCount() const { return static_cast<uint32_t>(m_textures.size()); }

    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }
    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }

private:
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    std::vector<Texture> m_textures;
    std::vector<std::string> m_loadedPaths;

    // Placeholder 1x1 white texture for when textureIndex == -1
    Texture m_placeholder;

    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    bool createSampler(VkDevice device);
    bool createDescriptorSetLayout(VkDevice device);
    bool createDescriptorPool(VkDevice device);
    bool createDescriptorSet(VkDevice device);
    void updateDescriptorSet(VkDevice device);

    bool createTextureFromPixels(VulkanContext& context,
                                 const uint8_t* pixels, uint32_t width, uint32_t height,
                                 uint32_t channels, Texture& outTexture);

    bool createPlaceholderTexture(VulkanContext& context);

    uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);

    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkImageLayout oldLayout, VkImageLayout newLayout);
};