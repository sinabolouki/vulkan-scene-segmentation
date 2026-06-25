#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct VulkanContext;
class SwapchainManager;
struct SceneObject;

struct FragmentNode {
    uint32_t color;      // Packed RGBA8
    float    depth;      // Fragment depth [0, 1]
    uint32_t objectID;   // Which object this fragment belongs to
    uint32_t next;       // Index of next node (0xFFFFFFFF = null)
};

class PPLLRenderer {
public:
    static constexpr uint32_t AVG_FRAGS_PER_PIXEL = 8;
    static constexpr uint32_t MAX_FRAGS_PER_PIXEL = 32;
    static constexpr uint32_t PPLL_NULL = 0xFFFFFFFF;

    bool init(VulkanContext& context, SwapchainManager& swapchain);
    void destroy(VkDevice device);
    void recreate(VulkanContext& context, SwapchainManager& swapchain);

    void recordCommands(VkCommandBuffer cmd,
                        VulkanContext& context,
                        SwapchainManager& swapchain,
                        VkBuffer vertexBuffer,
                        VkBuffer indexBuffer,
                        const std::vector<SceneObject>& objects,
                        const glm::mat4& viewProj,
                        uint32_t swapchainImageIndex);

    uint32_t getNodeBufferCapacity() const { return m_nodeCapacity; }

private:
    bool createHeadPointerImage(VulkanContext& context, uint32_t width, uint32_t height);
    bool createFragmentNodeBuffer(VulkanContext& context, uint32_t width, uint32_t height);
    bool createAtomicCounterBuffer(VulkanContext& context);
    bool createRenderPasses(VkDevice device, VkFormat swapchainFormat);
    bool createDescriptorSetLayouts(VkDevice device);
    bool createDescriptorPool(VkDevice device);
    bool createDescriptorSets(VkDevice device);
    bool createPipelineLayouts(VkDevice device);
    bool createPipelines(VkDevice device);
    bool createFramebuffers(VkDevice device, SwapchainManager& swapchain);

    void recordClearPhase(VkCommandBuffer cmd);
    void recordCollectPhase(VkCommandBuffer cmd,
                            VkBuffer vertexBuffer,
                            VkBuffer indexBuffer,
                            const std::vector<SceneObject>& objects,
                            const glm::mat4& viewProj,
                            VkExtent2D extent);
    void recordBarrier(VkCommandBuffer cmd);
    void recordResolvePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex, VkExtent2D extent);

    uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);
    void destroyDataStructures(VkDevice device);

    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Data structures
    VkImage m_headPointerImage = VK_NULL_HANDLE;
    VkDeviceMemory m_headPointerMemory = VK_NULL_HANDLE;
    VkImageView m_headPointerView = VK_NULL_HANDLE;

    VkBuffer m_nodeBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_nodeBufferMemory = VK_NULL_HANDLE;
    uint32_t m_nodeCapacity = 0;

    VkBuffer m_atomicCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_atomicCounterMemory = VK_NULL_HANDLE;

    // Render passes
    VkRenderPass m_collectRenderPass = VK_NULL_HANDLE;
    VkRenderPass m_resolveRenderPass = VK_NULL_HANDLE;

    // Framebuffers
    VkFramebuffer m_collectFramebuffer = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_resolveFramebuffers;

    // Descriptors
    VkDescriptorSetLayout m_ppllDescLayout = VK_NULL_HANDLE;  // Shared: head image + SSBO + counter
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_ppllDescSet = VK_NULL_HANDLE;           // Both passes use same set

    // Pipeline layouts
    VkPipelineLayout m_collectPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_resolvePipelineLayout = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline m_collectPipeline = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;
};