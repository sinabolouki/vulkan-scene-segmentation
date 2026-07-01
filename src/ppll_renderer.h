#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct VulkanContext;
class SwapchainManager;
struct SceneObject;

// GPU-side fragment node — must match GLSL layout exactly (16 bytes)
struct FragmentNode {
    uint32_t color;      // Packed RGBA8
    float    depth;      // Fragment depth [0, 1]
    uint32_t objectID;   // Which object this fragment belongs to
    uint32_t next;       // Index of next node (0xFFFFFFFF = null)
};

class PPLLRenderer {
public:
    // ─── Configuration ───
    static constexpr uint32_t AVG_FRAGS_PER_PIXEL = 8;
    static constexpr uint32_t MAX_FRAGS_PER_PIXEL = 32;
    static constexpr uint32_t NUM_PEEL_LAYERS = 4;
    static constexpr uint32_t PPLL_NULL = 0xFFFFFFFF;

    // Adjustable at runtime
    float depthThreshold = 0.02f;
    int debugDisplayLayer = -1; // -1 = off, 0..NUM_PEEL_LAYERS-1 = show that layer

    // ─── Public Interface ───
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

    // ─── Debug Info ───
    uint32_t getNodeBufferCapacity() const { return m_nodeCapacity; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    // ═══════════════════════════════════════════════════════════
    // CREATION FUNCTIONS
    // ═══════════════════════════════════════════════════════════

    bool createDataStructures(VulkanContext& context);
    bool createLayerImages(VulkanContext& context);
    bool createRenderPasses(VkDevice device, VkFormat swapchainFormat);
    bool createFramebuffers(VkDevice device, SwapchainManager& swapchain);
    bool createSampler(VkDevice device);
    bool createDescriptorSetLayouts(VkDevice device);
    bool createDescriptorPool(VkDevice device);
    bool createDescriptorSets(VkDevice device);
    bool createPipelineLayouts(VkDevice device);
    bool createPipelines(VkDevice device);

    // ═══════════════════════════════════════════════════════════
    // COMMAND RECORDING PHASES
    // ═══════════════════════════════════════════════════════════

    // Phase 1: Reset head pointers, counter, state
    void recordClearPhase(VkCommandBuffer cmd);

    // Phase 2: Rasterize geometry → insert all fragments into PPLL
    void recordCollectPhase(VkCommandBuffer cmd,
                            VkBuffer vertexBuffer, VkBuffer indexBuffer,
                            const std::vector<SceneObject>& objects,
                            const glm::mat4& viewProj);

    // Phase 3: Sort each pixel's linked list front-to-back
    void recordSortPhase(VkCommandBuffer cmd);

    // Phase 4 (per layer): Find global minimum depth among unconsumed fragments
    void recordFindMinPhase(VkCommandBuffer cmd);

    // Phase 5 (per layer): Resolve — composite fragments in depth range
    void recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex);

    // Phase 6 (per layer): Update state — consumedDepth = globalMin + threshold
    void recordUpdateStatePhase(VkCommandBuffer cmd);

    // Phase 7: Blend all layers back-to-front onto swapchain
    void recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex);

    // Generic memory barrier helper
    void recordMemoryBarrier(VkCommandBuffer cmd,
                             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                             VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    // ═══════════════════════════════════════════════════════════
    // HELPER FUNCTIONS
    // ═══════════════════════════════════════════════════════════

    bool createImage(VulkanContext& context, uint32_t w, uint32_t h,
                     VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                     VkImage& image, VkDeviceMemory& memory, VkImageView& view);

    bool createBuffer(VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkBuffer& buffer, VkDeviceMemory& memory);

    uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                            VkMemoryPropertyFlags properties);

    void destroySizeDependent(VkDevice device);

    // ═══════════════════════════════════════════════════════════
    // DIMENSIONS
    // ═══════════════════════════════════════════════════════════

    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // ═══════════════════════════════════════════════════════════
    // PPLL DATA STRUCTURES
    // ═══════════════════════════════════════════════════════════

    // Head pointer image (R32_UINT): each pixel → first node index
    VkImage        m_headPointerImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_headPointerMemory = VK_NULL_HANDLE;
    VkImageView    m_headPointerView   = VK_NULL_HANDLE;

    // Fragment node buffer (SSBO): array of FragmentNode
    VkBuffer       m_nodeBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_nodeBufferMemory = VK_NULL_HANDLE;
    uint32_t       m_nodeCapacity     = 0;

    // Atomic counter (SSBO, 4 bytes): next free slot
    VkBuffer       m_atomicCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_atomicCounterMemory = VK_NULL_HANDLE;

    // State buffer (SSBO, 8 bytes): { uint globalMinUint, uint consumedDepthBits }
    VkBuffer       m_stateBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_stateBufferMemory = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // LAYER COLOR IMAGES (one per peel layer)
    // ═══════════════════════════════════════════════════════════

    struct LayerImage {
        VkImage        image       = VK_NULL_HANDLE;
        VkDeviceMemory memory      = VK_NULL_HANDLE;
        VkImageView    view        = VK_NULL_HANDLE;
        VkFramebuffer  framebuffer = VK_NULL_HANDLE; // For resolve output
    };
    std::vector<LayerImage> m_layerImages;

    // ═══════════════════════════════════════════════════════════
    // RENDER PASSES
    // ═══════════════════════════════════════════════════════════

    // 0 attachments: used by collect, sort, find-min, update-state
    VkRenderPass m_noAttachmentRenderPass = VK_NULL_HANDLE;

    // 1 color attachment (RGBA8): used by resolve → layer texture
    VkRenderPass m_layerRenderPass = VK_NULL_HANDLE;

    // 1 color attachment (swapchain format): used by composite
    VkRenderPass m_compositeRenderPass = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // FRAMEBUFFERS
    // ═══════════════════════════════════════════════════════════

    // For collect/sort/find-min/update-state (no attachments)
    VkFramebuffer m_noAttachmentFramebuffer = VK_NULL_HANDLE;

    // For composite (one per swapchain image)
    std::vector<VkFramebuffer> m_compositeFramebuffers;

    // Layer framebuffers are stored inside LayerImage::framebuffer

    // ═══════════════════════════════════════════════════════════
    // SAMPLER (for composite pass to read layer textures)
    // ═══════════════════════════════════════════════════════════

    VkSampler m_sampler = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // DESCRIPTOR LAYOUTS
    // ═══════════════════════════════════════════════════════════

    // Shared PPLL set: head image (storage) + nodes (SSBO) + counter (SSBO) + state (SSBO)
    VkDescriptorSetLayout m_ppllDescLayout = VK_NULL_HANDLE;

    // Composite set: layer textures as combined image samplers
    VkDescriptorSetLayout m_compositeDescLayout = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // DESCRIPTOR POOL & SETS
    // ═══════════════════════════════════════════════════════════

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_ppllDescSet      = VK_NULL_HANDLE; // Used by collect, sort, find-min, resolve, update
    VkDescriptorSet  m_compositeDescSet = VK_NULL_HANDLE; // Used by composite

    // ═══════════════════════════════════════════════════════════
    // PIPELINE LAYOUTS
    // ═══════════════════════════════════════════════════════════

    // Collect: set 0 = ppll, push = ScenePushConstants (model, viewProj, objectID)
    VkPipelineLayout m_collectPipelineLayout = VK_NULL_HANDLE;

    // Fullscreen passes (sort, find-min, resolve, update-state):
    //   set 0 = ppll, push = { float depthThreshold }
    VkPipelineLayout m_fullscreenPipelineLayout = VK_NULL_HANDLE;

    // Composite: set 0 = composite textures, push = { uint numLayers }
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // PIPELINES
    // ═══════════════════════════════════════════════════════════

    VkPipeline m_collectPipeline     = VK_NULL_HANDLE; // Geometry → SSBO (no color, no depth test)
    VkPipeline m_sortPipeline        = VK_NULL_HANDLE; // Fullscreen → rewrite next ptrs (no color)
    VkPipeline m_findMinPipeline     = VK_NULL_HANDLE; // Fullscreen → atomicMin to state (no color)
    VkPipeline m_updateStatePipeline = VK_NULL_HANDLE; // Fullscreen → update consumed depth (no color)
    VkPipeline m_resolvePipeline     = VK_NULL_HANDLE; // Fullscreen → layer color output
    VkPipeline m_compositePipeline   = VK_NULL_HANDLE; // Fullscreen → blend layers to swapchain
};