#pragma once

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>
#include <cstdint>

struct VulkanContext; // forward declaration

class SwapchainManager {
public:
    bool init(VulkanContext& context, uint32_t width, uint32_t height);
    void destroy(VkDevice device);
    bool recreate(VulkanContext& context, uint32_t width, uint32_t height);

    // Frame operations
    VkResult acquireNextImage(VkDevice device, uint64_t timeout, uint32_t& imageIndex);
    VkResult present(VkQueue presentQueue, uint32_t imageIndex, VkSemaphore waitSemaphore);

    // Getters
    VkSwapchainKHR getSwapchain() const { return m_swapchain; }
    VkRenderPass getRenderPass() const { return m_renderPass; }
    VkFramebuffer getFramebuffer(uint32_t index) const { return m_framebuffers[index]; }
    VkExtent2D getExtent() const { return m_extent; }
    VkFormat getImageFormat() const { return m_imageFormat; }
    VkFormat getDepthFormat() const { return m_depthFormat; }
    uint32_t getImageCount() const { return static_cast<uint32_t>(m_images.size()); }
    VkImageView getImageView(uint32_t index) const { return m_imageViews[index]; }
    VkImage getImage(uint32_t index) const { return m_images[index]; }

    // Sync objects (per frame-in-flight)
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    VkSemaphore getImageAvailableSemaphore(uint32_t frame) const { return m_imageAvailableSemaphores[frame]; }
    VkSemaphore getRenderFinishedSemaphore(uint32_t frame) const { return m_renderFinishedSemaphores[frame]; }
    VkFence getInFlightFence(uint32_t frame) const { return m_inFlightFences[frame]; }

private:
    bool createSwapchain(VulkanContext& context, uint32_t width, uint32_t height);
    bool createImageViews(VkDevice device);
    bool createDepthResources(VulkanContext& context);
    bool createRenderPass(VkDevice device);
    bool createFramebuffers(VkDevice device);
    bool createSyncObjects(VkDevice device);

    void destroySwapchainResources(VkDevice device);

    VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
    VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                                const std::vector<VkFormat>& candidates,
                                VkImageTiling tiling,
                                VkFormatFeatureFlags features);

    // Swapchain
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent = {0, 0};
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;

    // Depth
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkImage m_depthImage = VK_NULL_HANDLE;
    VkDeviceMemory m_depthImageMemory = VK_NULL_HANDLE;
    VkImageView m_depthImageView = VK_NULL_HANDLE;

    // Render pass & framebuffers
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;

    // Synchronization
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
};