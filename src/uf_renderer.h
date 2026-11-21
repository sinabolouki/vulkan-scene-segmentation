#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <string>

#include "texture_manager.h"
#include "uf_evaluation.h"
#include "performance_capture.h"
#include <array>

struct VulkanContext;
class SwapchainManager;
struct SceneObject;

// GPU-side fragment node — must match GLSL layout exactly (32 bytes)
struct UFFragmentNode {
    uint32_t color;      // Packed RGBA8
    uint32_t unshadedColor; // Packed RGBA8 (unshaded, for color-based merging)
    float    depth;      // Fragment depth [0, 1] linear
    float    normalX;    // View-space normal X
    float    normalY;    // View-space normal Y
    float    normalZ;    // View-space normal Z
    uint32_t objectID;   // Which object this fragment belongs to
    uint32_t next;       // Index of next node (0xFFFFFFFF = null)
};


class UFRenderer {
public:
    // ─── Configuration ───
    static constexpr uint32_t AVG_FRAGS_PER_PIXEL = 16;
    static constexpr uint32_t MAX_FRAGS_PER_PIXEL = 32;
    static constexpr uint32_t NUM_LAYERS = 16;
    static constexpr uint32_t UF_NULL = 0xFFFFFFFF;
    static constexpr uint32_t MAX_GLOBAL_ROOTS = 64;
    static constexpr uint32_t MAX_TIMESTAMPS = 64;

    // Runtime parameters (adjustable via ImGui)
    float planeThreshold = 1.5f;
    float proximityScale = 0.7f;
    float colorThreshold = 0.05f;
    int debugDisplayLayer = -1;
    uint32_t mergeMode = 0;  // 0 = balanced, 1 = color priority, 2 = space priority


    int debugVisualizationMode = 4;  // -1 = normal peeling, 0-8 = debug visualization modes

    struct RegionSelection {
        int32_t pixelX = -1;
        int32_t pixelY = -1;
        bool active = false;
    };

    RegionSelection regionSelection;

    void clearRegionSelection()
    {
        regionSelection = RegionSelection{};
    }

    void selectRegionAtPixel(int32_t x, int32_t y);

    // One-shot CSV evaluation capture.
    bool requestEvaluationExport(std::string filename);
    bool isEvaluationExportPending() const
    {
        return !m_evaluationPath.empty();
    }
    const std::string& evaluationExportStatus() const
    {
        return m_evaluationStatus;
    }
    // Returns true when an export frame was processed.
    bool finishEvaluationExport(VkDevice device, VkFence submittedFence);



    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    bool bypassUnionFind = false;

    struct TimingResults {
        float clear = 0;
        float collect = 0;
        float sort = 0;
        float initParents = 0;
        float grouping = 0; // All merge-compression iterations and their barriers.
        float peelTotal = 0;
        struct PeelTiming {
            float clearTable = 0;
            float registerRoots = 0;
            float resolve = 0;
            float total = 0;
        };
        PeelTiming peels[NUM_LAYERS];
        float composite = 0;
        float total = 0;
    };
    TimingResults timingResults;
    bool enableTiming = true;

    struct DebugStats {
        uint32_t fragmentCount = 0;
        uint32_t nodeCapacity = 0;
        bool     overflowed = false;
        float    usage = 0.0f;  // percentage
    };
    DebugStats debugStats;

    struct DebugGPUInfo {
        uint32_t minMaxPos = 0;
        uint32_t numRegisteredRoots = 0;
        uint32_t pad[2];  // Pad to 16 bytes for alignment
    };
    DebugGPUInfo debugGPUInfo;

    bool isTimingReady() const { return m_timingReady; }
    // Call only after this flight slot's submission fence has signalled.
    // No query WAIT flag, no extra fence/device/queue waits.
    std::optional<performance::GpuFrameResult> resolveCompletedFrame(VkDevice device, uint32_t slot);
    void frameSubmitted(uint32_t slot) { m_profileFrames.at(slot).pending = true; }
    bool timestampsSupported() const { return m_timestampQueryPool != VK_NULL_HANDLE; }
    uint32_t timestampValidBits() const { return m_timestampValidBits; }
    double timestampPeriodNs() const { return m_timestampPeriod; }

    // Exclude the heavyweight, synchronous evaluation export while benchmarking.
    void setPerformanceCaptureActive(bool active) { m_performanceCaptureActive = active; }

    // ─── Texture Manager ───
    void setTextureManager(TextureManager* texMgr) { m_textureManager = texMgr; }

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
                        const glm::mat4& view,
                        const glm::mat4& projection,
                        uint32_t swapchainImageIndex,
                        uint32_t frameSlot, uint64_t applicationFrameId);

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

    void recordClearPhase(VkCommandBuffer cmd);

    void recordCollectPhase(VkCommandBuffer cmd,
                            VkBuffer vertexBuffer, VkBuffer indexBuffer,
                            const std::vector<SceneObject>& objects,
                            const glm::mat4& viewProj, const glm::mat4& view);

    void recordSortPhase(VkCommandBuffer cmd);
    void recordInitParentsPhase(VkCommandBuffer cmd);
    void recordMergePhase(VkCommandBuffer cmd);
    void recordCompressPhase(VkCommandBuffer cmd);
    void recordRegisterRootsPhase(VkCommandBuffer cmd);
    void recordFindMinPhase(VkCommandBuffer cmd);
    void recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex);
    void recordCompositePhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex);
    void recordDebugRegionsPhase(VkCommandBuffer cmd, uint32_t swapchainImageIndex);
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

    bool createProfilingResources(VulkanContext& context);
    void destroyProfilingResources(VkDevice device);
    void destroySizeDependent(VkDevice device);

    struct EvaluationReadback {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize headsOffset = 0;
        VkDeviceSize parentsOffset = 0;
        VkDeviceSize nodesOffset = 0;
        VkDeviceSize bytes = 0;
        bool coherent = false;
        bool recorded = false;
        uf_eval::Metadata metadata;
    };

    EvaluationReadback m_evaluationReadback;
    std::string m_evaluationPath;
    std::string m_evaluationStatus = "No evaluation export yet.";

    bool prepareEvaluationReadback(VulkanContext& context);
    void recordEvaluationReadback(VkCommandBuffer cmd);
    void destroyEvaluationReadback(VkDevice device);


    // ═══════════════════════════════════════════════════════════
    // DIMENSIONS
    // ═══════════════════════════════════════════════════════════

    uint32_t m_width = 0;
    uint32_t m_height = 0;

    float m_projX = 1.0f;
    float m_projY = 1.0f;

    // ═══════════════════════════════════════════════════════════
    // TEXTURE MANAGER (externally owned)
    // ═══════════════════════════════════════════════════════════

    TextureManager* m_textureManager = nullptr;

    // ═══════════════════════════════════════════════════════════
    // PPLL DATA STRUCTURES
    // ═══════════════════════════════════════════════════════════

    VkImage        m_headPointerImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_headPointerMemory = VK_NULL_HANDLE;
    VkImageView    m_headPointerView   = VK_NULL_HANDLE;

    VkBuffer       m_nodeBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_nodeBufferMemory = VK_NULL_HANDLE;
    uint32_t       m_nodeCapacity     = 0;

    VkBuffer       m_atomicCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_atomicCounterMemory = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // UNION-FIND DATA STRUCTURE
    // ═══════════════════════════════════════════════════════════

    VkBuffer       m_parentBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_parentBufferMemory = VK_NULL_HANDLE;

    VkBuffer       m_rootMaxPosBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_rootMaxPosMemory       = VK_NULL_HANDLE;

    VkBuffer       m_globalMinBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory m_globalMinMemory        = VK_NULL_HANDLE;

    VkBuffer       m_globalMinStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_globalMinStagingMemory = VK_NULL_HANDLE;
    void*          m_globalMinStagingMapped = nullptr;

    // Pipeline for the find-min pass
    VkPipeline     m_findMinPipeline        = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // LAYER COLOR IMAGES
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

    VkRenderPass m_noAttachmentRenderPass = VK_NULL_HANDLE;
    VkRenderPass m_layerRenderPass = VK_NULL_HANDLE;
    VkRenderPass m_compositeRenderPass = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // FRAMEBUFFERS
    // ═══════════════════════════════════════════════════════════

    VkFramebuffer m_noAttachmentFramebuffer = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_compositeFramebuffers;

    // ═══════════════════════════════════════════════════════════
    // SAMPLER
    // ═══════════════════════════════════════════════════════════

    VkSampler m_sampler = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // DESCRIPTOR LAYOUTS
    // ═══════════════════════════════════════════════════════════

    VkDescriptorSetLayout m_ppllDescLayout = VK_NULL_HANDLE;
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

    VkPipelineLayout m_collectPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_fullscreenPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_compositePipelineLayout = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // PIPELINES
    // ═══════════════════════════════════════════════════════════

    VkPipeline m_collectPipeline     = VK_NULL_HANDLE;
    VkPipeline m_sortPipeline        = VK_NULL_HANDLE;
    VkPipeline m_initParentsPipeline = VK_NULL_HANDLE;
    VkPipeline m_mergePipeline       = VK_NULL_HANDLE;
    VkPipeline m_compressPipeline    = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline     = VK_NULL_HANDLE;
    VkPipeline m_compositePipeline   = VK_NULL_HANDLE;
    VkPipeline m_registerRootsPipeline = VK_NULL_HANDLE;
    VkPipeline m_debugRegionsPipeline = VK_NULL_HANDLE;

    // ═══════════════════════════════════════════════════════════
    // TIMESTAMPING
    // ═══════════════════════════════════════════════════════════

    struct ProfileFrame {
        VkBuffer readbackBuffer = VK_NULL_HANDLE;
        VkDeviceMemory readbackMemory = VK_NULL_HANDLE;
        void* mapped = nullptr; // Host-visible, coherent; read only after fence.
        uint64_t frameId = 0;
        uint32_t nodeCapacity = 0;
        uint32_t timestampCount = 0;
        bool pending = false;
        bool timed = false;
        bool layered = false;
        bool grouping = false;
        bool evaluationExport = false;
    };
    bool m_performanceCaptureActive = false;
    std::vector<ProfileFrame> m_profileFrames; // One per normal frame in flight.
    VkQueryPool m_timestampQueryPool = VK_NULL_HANDLE; // Disjoint slot ranges.
    double m_timestampPeriod = 0.0;
    uint32_t m_timestampValidBits = 0;
    bool m_timingReady = false;

    VkBuffer       m_debugInfoBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory m_debugInfoMemory        = VK_NULL_HANDLE;
};