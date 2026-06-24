#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "scene.h"

struct VulkanContext;
class SwapchainManager;

// One peeling layer: color + depth + objectID
struct PeelLayer {
    VkImage colorImage = VK_NULL_HANDLE;
    VkDeviceMemory colorMemory = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE;

    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

class DepthPeelingRenderer {
public:
    static constexpr uint32_t NUM_PEEL_LAYERS = 4; // Configurable

    bool init(VulkanContext& context, SwapchainManager& swapchain);
    void destroy(VkDevice device);
    void recreate(VulkanContext& context, SwapchainManager& swapchain);

    // Record all depth peeling passes into the command buffer
    void recordCommands(VkCommandBuffer cmd,
                    VulkanContext& context,
                    SwapchainManager& swapchain,
                    VkBuffer vertexBuffer,
                    VkBuffer indexBuffer,
                    const std::vector<SceneObject>& objects,  // ← Changed
                    VkPipelineLayout sceneLayout,
                    const glm::mat4& viewProj,
                    uint32_t swapchainImageIndex);

    // Getters
    VkRenderPass getPeelRenderPass() const { return m_peelRenderPass; }
    VkRenderPass getCompositeRenderPass() const { return m_compositeRenderPass; }
    VkPipeline getInitPipeline() const { return m_initPipeline; }
    VkPipeline getPeelPipeline() const { return m_peelPipeline; }
    VkPipeline getCompositePipeline() const { return m_compositePipeline; }
    VkPipelineLayout getScenePipelineLayout() const { return m_scenePipelineLayout; }
    VkPipelineLayout getCompositePipelineLayout() const { return m_compositePipelineLayout; }

private:
    // Setup functions
    bool createPeelLayers(VulkanContext& context, uint32_t width, uint32_t height);
    bool createRenderPasses(VkDevice device, VkFormat colorFormat, VkFormat depthFormat);
    bool createFramebuffers(VkDevice device, uint32_t width, uint32_t height);
    bool createSampler(VkDevice device);
    bool createDescriptorSetLayout(VkDevice device);
    bool createDescriptorPool(VkDevice device);
    bool createDescriptorSets(VkDevice device);
    bool createPipelineLayouts(VkDevice device);
    bool createPipelines(VkDevice device, VkRenderPass swapchainRenderPass);
    bool createCompositeFramebuffers(VkDevice device, SwapchainManager& swapchain);

    // Helpers
    bool createImage(VulkanContext& context, uint32_t width, uint32_t height,
                     VkFormat format, VkImageUsageFlags usage,
                     VkImageAspectFlags aspect,
                     VkImage& image, VkDeviceMemory& memory, VkImageView& view);
    uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);
    void destroyPeelResources(VkDevice device);

    // Layers
    std::vector<PeelLayer> m_layers;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;

    // Render passes
    VkRenderPass m_peelRenderPass = VK_NULL_HANDLE;     // Used for all peel layers
    VkRenderPass m_compositeRenderPass = VK_NULL_HANDLE;

    // Composite framebuffers (one per swapchain image)
    std::vector<VkFramebuffer> m_compositeFramebuffers;

    // Sampler for reading previous depth
    VkSampler m_depthSampler = VK_NULL_HANDLE;
    VkSampler m_colorSampler = VK_NULL_HANDLE;

    // Descriptors
    VkDescriptorSetLayout m_peelDescriptorSetLayout = VK_NULL_HANDLE;    // prev depth
    VkDescriptorSetLayout m_compositeDescriptorSetLayout = VK_NULL_HANDLE; // all color layers
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    VkDescriptorSet m_dummyDescriptorSet = VK_NULL_HANDLE; // For layer 0 (no previous depth)

    // Per-layer descriptor set (binds previous layer's depth)
    std::vector<VkDescriptorSet> m_peelDescriptorSets; // [layer 1..N-1]

    // Composite descriptor set (binds all color layers)
    VkDescriptorSet m_compositeDescriptorSet = VK_NULL_HANDLE;

    // Pipeline layouts
    VkPipelineLayout m_scenePipelineLayout = VK_NULL_HANDLE;     // push constants + prev depth
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE; // color layer samplers

    // Pipelines
    VkPipeline m_initPipeline = VK_NULL_HANDLE;
    VkPipeline m_peelPipeline = VK_NULL_HANDLE;
    VkPipeline m_compositePipeline = VK_NULL_HANDLE;
};