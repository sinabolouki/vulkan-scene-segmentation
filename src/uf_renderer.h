#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct VulkanContext;
class SwapchainManager;
struct SceneObject;

// GPU-side fragment node — must match GLSL layout exactly (20 bytes)
struct UFFragmentNode {
    uint32_t color;      // Packed RGBA8
    float    depth;      // Fragment depth [0, 1] linear
    uint32_t normalXY;   // ← NEW: packed view-space normal (2x float16)
    uint32_t objectID;   // Which object this fragment belongs to
    uint32_t next;       // Index of next node (0xFFFFFFFF = null)
};
// sizeof = 20 bytes (will be padded to 20 by std430 — no issue since it's array of scalars)

class UFRenderer {
public:
    // ─── Configuration ───
    static constexpr uint32_t AVG_FRAGS_PER_PIXEL = 8;
    static constexpr uint32_t MAX_FRAGS_PER_PIXEL = 32;
    static constexpr uint32_t NUM_LAYERS = 8;
    static constexpr uint32_t UF_NULL = 0xFFFFFFFF;
    static constexpr uint32_t MAX_GLOBAL_ROOTS = 32;
    static constexpr uint32_t MAX_TIMESTAMPS = 40;

    // Runtime parameters (adjustable via ImGui)
    float depthThreshold = 0.007f;
    float colorThreshold = 1.0f;
    int debugDisplayLayer = -1;   // -1 = composite all, 0-3 = single layer

    float nearPlane = 0.1f;
    float farPlane = 100.0f;

     struct TimingResults {
        float clear = 0;
        float collect = 0;
        float sort = 0;
        float initParents = 0;
        float merge = 0;
        float compress = 0;
        float peelTotal = 0;
        struct PeelTiming {
            float clearTable = 0;
            float registerRoots = 0;
            float resolve = 0;
            float total = 0;
        };
        PeelTiming peels[NUM_LAYERS];  // per-peel (register + resolve)
        float composite = 0;
        float total = 0;
    };
    TimingResults timingResults;
    bool enableTiming = true;

    bool isTimingReady() const { return m_timingReady; }
    void readTimingResults(VkDevice device);

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
                        const glm::mat4& view,        // ← NEW
                        const glm::mat4& projection,  // ← NEW (needed for projX, projY)
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

    // Phase 1: Reset head pointers and atomic counter
    void recordClearPhase(VkCommandBuffer cmd);

    // Phase 2: Rasterize geometry → insert all fragments into PPLL
    void recordCollectPhase(VkCommandBuffer cmd,
                            VkBuffer vertexBuffer, VkBuffer indexBuffer,
                            const std::vector<SceneObject>& objects,
                            const glm::mat4& viewProj, const glm::mat4& view);

    // Phase 3: Sort each pixel's linked list front-to-back
    void recordSortPhase(VkCommandBuffer cmd);

    // Phase 4: Initialize parent[i] = i for all allocated fragments
    void recordInitParentsPhase(VkCommandBuffer cmd);

    // Phase 5: For each fragment, compare with all neighbor fragments → Union()
    void recordMergePhase(VkCommandBuffer cmd);

    // Phase 6: Path compression — parent[i] = root of tree containing i
    void recordCompressPhase(VkCommandBuffer cmd);

    // Phase 6.5: Register all unique roots with their max depth
    void recordRegisterRootsPhase(VkCommandBuffer cmd);

    // Phase 7 (per layer): Walk sorted list, output N-th distinct root's color
    void recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex);

    // Phase 8: Blend all layers back-to-front onto swapchain
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

    void computeTimingResults();

    void destroySizeDependent(VkDevice device);

    // ═══════════════════════════════════════════════════════════
    // DIMENSIONS
    // ═══════════════════════════════════════════════════════════

    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Projection matrix parameters (for reconstructing view-space position)
    float m_projX = 1.0f;
    float m_projY = 1.0f;

    // ═══════════════════════════════════════════════════════════
    // PPLL DATA STRUCTURES
    // ═══════════════════════════════════════════════════════════

    // Head pointer image (R32_UINT): each pixel → first node index
    VkImage        m_headPointerImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_headPointerMemory = VK_NULL_HANDLE;
    VkImageView    m_headPointerView   = VK_NULL_HANDLE;

    // Fragment node buffer (SSBO): array of UFFragmentNode
    VkBuffer       m_nodeBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_nodeBufferMemory = VK_NULL_HANDLE;
    uint32_t       m_nodeCapacity     = 0;

    // Atomic counter (SSBO, 4 bytes): next free slot in node buffer
    VkBuffer       m_atomicCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_atomicCounterMemory = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // UNION-FIND DATA STRUCTURE
    // ═══════════════════════════════════════════════════════════

    // Parent buffer (SSBO): parent[i] = parent index of fragment i
    // After compress: parent[i] = root of connected component
    VkBuffer       m_parentBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_parentBufferMemory = VK_NULL_HANDLE;

    // Global root table (SSBO): 8 entries × {rootID, maxDepth} = 64 bytes
    VkBuffer       m_rootTableBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_rootTableBufferMemory = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // LAYER COLOR IMAGES (one per output layer)
    // ═══════════════════════════════════════════════════════════

    struct LayerImage {
        VkImage        image       = VK_NULL_HANDLE;
        VkDeviceMemory memory      = VK_NULL_HANDLE;
        VkImageView    view        = VK_NULL_HANDLE;
        VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    };
    std::vector<LayerImage> m_layerImages;

    // ═══════════════════════════════════════════════════════════
    // RENDER PASSES
    // ═══════════════════════════════════════════════════════════

    // No attachments: collect, sort, init_parents, merge, compress
    VkRenderPass m_noAttachmentRenderPass = VK_NULL_HANDLE;

    // 1 color attachment (RGBA8): resolve → layer texture
    VkRenderPass m_layerRenderPass = VK_NULL_HANDLE;

    // 1 color attachment (swapchain format): composite → swapchain
    VkRenderPass m_compositeRenderPass = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // FRAMEBUFFERS
    // ═══════════════════════════════════════════════════════════

    VkFramebuffer m_noAttachmentFramebuffer = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_compositeFramebuffers;
    // Layer framebuffers stored inside LayerImage::framebuffer

    // ═══════════════════════════════════════════════════════════
    // SAMPLER
    // ═══════════════════════════════════════════════════════════

    VkSampler m_sampler = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // DESCRIPTOR LAYOUTS
    // ═══════════════════════════════════════════════════════════

    // PPLL set (set 0 for collect/sort/init/merge/compress/resolve):
    //   binding 0: head pointer image    (storage image, r32ui)
    //   binding 1: node buffer           (storage buffer)
    //   binding 2: atomic counter        (storage buffer)
    //   binding 3: parent buffer         (storage buffer)
    VkDescriptorSetLayout m_ppllDescLayout = VK_NULL_HANDLE;

    // Composite set (set 0 for composite pipeline):
    //   binding 0..N-1: layer color textures (combined image sampler)
    VkDescriptorSetLayout m_compositeDescLayout = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // DESCRIPTOR POOL & SETS
    // ═══════════════════════════════════════════════════════════

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_ppllDescSet      = VK_NULL_HANDLE;
    VkDescriptorSet  m_compositeDescSet = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // PIPELINE LAYOUTS
    // ═══════════════════════════════════════════════════════════

    // Collect: set 0 = ppll, push = ScenePushConstants
    VkPipelineLayout m_collectPipelineLayout = VK_NULL_HANDLE;

    // Fullscreen passes (sort, init, merge, compress, resolve):
    //   set 0 = ppll, push = UFFullscreenPushConstants
    VkPipelineLayout m_fullscreenPipelineLayout = VK_NULL_HANDLE;

    // Composite: set 0 = composite textures, push = { uint numLayers, int displayLayer }
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // PIPELINES
    // ═══════════════════════════════════════════════════════════

    VkPipeline m_collectPipeline     = VK_NULL_HANDLE;  // Geometry → PPLL
    VkPipeline m_sortPipeline        = VK_NULL_HANDLE;  // Sort linked lists
    VkPipeline m_initParentsPipeline = VK_NULL_HANDLE;  // parent[i] = i
    VkPipeline m_mergePipeline       = VK_NULL_HANDLE;  // Union neighbors
    VkPipeline m_compressPipeline    = VK_NULL_HANDLE;  // Path compression
    VkPipeline m_resolvePipeline     = VK_NULL_HANDLE;  // Root → layer color
    VkPipeline m_compositePipeline   = VK_NULL_HANDLE;  // Blend layers
    VkPipeline m_registerRootsPipeline = VK_NULL_HANDLE;  // Register global roots

    //TIMESTAMPING
    VkQueryPool m_timestampQueryPool = VK_NULL_HANDLE;
    float       m_timestampPeriod = 0.0f;  // nanoseconds per tick
    uint64_t    m_timestamps[MAX_TIMESTAMPS] = {};
    uint32_t    m_timestampCount = 0;
    bool        m_timingReady = false;
};