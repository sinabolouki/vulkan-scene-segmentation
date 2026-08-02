#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct VulkanContext;
class SwapchainManager;
class TextureManager;
struct SceneObject;

// Push constants for the simple forward renderer
struct SimpleScenePushConstants {
    glm::mat4 model;       // 64 bytes
    glm::mat4 viewProj;    // 64 bytes
    glm::mat4 view;        // 64 bytes
    int32_t   textureIndex;// 4 bytes (-1 = no texture)
    float     pad[3];      // 12 bytes padding to align to 16
};
// Total: 208 bytes

class SimpleRenderer {
public:
    // ─── Runtime Parameters ───
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    // ─── Public Interface ───
    bool init(VulkanContext& context, SwapchainManager& swapchain);
    void destroy(VkDevice device);
    void recreate(VulkanContext& context, SwapchainManager& swapchain);

    void setTextureManager(TextureManager* texMgr) { m_textureManager = texMgr; }

    void recordCommands(VkCommandBuffer cmd,
                        VulkanContext& context,
                        SwapchainManager& swapchain,
                        VkBuffer vertexBuffer,
                        VkBuffer indexBuffer,
                        const std::vector<SceneObject>& objects,
                        const glm::mat4& viewProj,
                        const glm::mat4& view,
                        const glm::mat4& projection,
                        uint32_t swapchainImageIndex);

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    // ─── Creation ───
    bool createDepthImage(VulkanContext& context);
    bool createRenderPass(VkDevice device, VkFormat swapchainFormat);
    bool createFramebuffers(VkDevice device, SwapchainManager& swapchain);
    bool createPipelineLayout(VkDevice device);
    bool createPipeline(VkDevice device);

    // ─── Helpers ───
    bool createImage(VulkanContext& context, uint32_t w, uint32_t h,
                     VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                     VkImage& image, VkDeviceMemory& memory, VkImageView& view);
    uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);
    void destroySizeDependent(VkDevice device);

    // ─── Dimensions ───
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // ─── Texture Manager (externally owned) ───
    TextureManager* m_textureManager = nullptr;

    // ─── Depth Image ───
    VkImage        m_depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_depthMemory = VK_NULL_HANDLE;
    VkImageView    m_depthView   = VK_NULL_HANDLE;

    // ─── Render Pass ───
    VkRenderPass m_renderPass = VK_NULL_HANDLE;

    // ─── Framebuffers (one per swapchain image) ───
    std::vector<VkFramebuffer> m_framebuffers;

    // ─── Pipeline ───
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline       = VK_NULL_HANDLE;
};