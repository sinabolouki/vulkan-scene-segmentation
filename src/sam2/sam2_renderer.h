#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <vector>
#include <cstdint>

#include "uf_renderer.h"   // UFFragmentNode + constants

struct VulkanContext;
struct SceneObject;
class  TextureManager;

class Sam2Renderer
{
public:
    struct CapturedLayer
    {
        uint32_t             width  = 0;
        uint32_t             height = 0;
        std::vector<uint8_t> rgba;   // width*height*4
    };

    struct Config
    {
        uint32_t width     = 1280;
        uint32_t height    = 720;
        uint32_t maxLayers = UFRenderer::NUM_LAYERS;
    };

    // Runtime tweakables
    float    planeThreshold = 1.5f;
    float    proximityScale = 0.7f;
    float    colorThreshold = 0.05f;
    float    nearPlane      = 0.1f;
    float    farPlane       = 100.0f;
    uint32_t mergeMode      = 0;

    void setTextureManager(TextureManager* mgr) { m_textureManager = mgr; }

    bool init(VulkanContext& ctx, const Config& cfg);
    void destroy(VkDevice device);

    // Rebuild size-dependent resources (call after swapchain recreation).
    void recreate(VulkanContext& ctx, uint32_t newWidth, uint32_t newHeight);

    void recordPreview(VkCommandBuffer cmd,
                       VkBuffer vertexBuffer,
                       VkBuffer indexBuffer,
                       const std::vector<SceneObject>& objects,
                       const glm::mat4& viewProj,
                       const glm::mat4& view,
                       const glm::mat4& projection);

    std::vector<CapturedLayer> captureSnapshot(VulkanContext& ctx,
                                               VkBuffer vertexBuffer,
                                               VkBuffer indexBuffer,
                                               const std::vector<SceneObject>& objects,
                                               const glm::mat4& viewProj,
                                               const glm::mat4& view,
                                               const glm::mat4& projection);

    VkImageView   previewImageView()   const { return m_layerImages.empty() ? VK_NULL_HANDLE : m_layerImages[0].view; }
    VkSampler     previewSampler()     const { return m_sampler; }
    VkImageLayout previewImageLayout() const { return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }

        // Blits layer 0 (preview) into a swapchain image.
    // Leaves the swapchain image in PRESENT_SRC_KHR (what ImGui's render pass expects).
    void recordBlitPreviewToSwapchain(VkCommandBuffer cmd,
                                      VkImage swapchainImage,
                                      VkExtent2D swapchainExtent);

    const Config& config() const { return m_cfg; }
    uint32_t getWidth()  const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    // ---- Creation --------------------------------------------------------
    bool createDataStructures(VulkanContext& ctx);
    bool createLayerImages(VulkanContext& ctx);
    bool createRenderPasses(VkDevice device);
    bool createFramebuffers(VkDevice device);
    bool createSampler(VkDevice device);
    bool createDescriptorSetLayout(VkDevice device);
    bool createDescriptorPool(VkDevice device);
    bool createDescriptorSet(VkDevice device);
    bool createPipelineLayouts(VkDevice device);
    bool createPipelines(VkDevice device);
    bool createStagingBuffer(VulkanContext& ctx);
    bool createCommandResources(VulkanContext& ctx);

    void destroySizeDependent(VkDevice device);

    // ---- Recording -------------------------------------------------------
    void recordClearPhase(VkCommandBuffer cmd);
    void recordCollectPhase(VkCommandBuffer cmd,
                            VkBuffer vertexBuffer, VkBuffer indexBuffer,
                            const std::vector<SceneObject>& objects,
                            const glm::mat4& viewProj, const glm::mat4& view);
    void recordSortPhase(VkCommandBuffer cmd);
    void recordResolvePhase(VkCommandBuffer cmd, uint32_t layerIndex);
    void recordCopyLayerToStaging(VkCommandBuffer cmd, uint32_t layerIndex);
    void recordMemoryBarrier(VkCommandBuffer cmd,
                             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                             VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    // ---- Helpers ---------------------------------------------------------
    bool createImage(VulkanContext& ctx, uint32_t w, uint32_t h,
                     VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                     VkImage& image, VkDeviceMemory& memory, VkImageView& view);
    bool createBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props);

    // ---- State -----------------------------------------------------------
    Config   m_cfg{};
    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    float    m_projX  = 1.0f;
    float    m_projY  = 1.0f;

    TextureManager* m_textureManager = nullptr;

    // PPLL
    VkImage        m_headPointerImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_headPointerMemory = VK_NULL_HANDLE;
    VkImageView    m_headPointerView   = VK_NULL_HANDLE;

    VkBuffer       m_nodeBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_nodeBufferMemory = VK_NULL_HANDLE;
    uint32_t       m_nodeCapacity     = 0;

    VkBuffer       m_atomicCounterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_atomicCounterMemory = VK_NULL_HANDLE;

    // Layer images
    struct LayerImage {
        VkImage        image       = VK_NULL_HANDLE;
        VkDeviceMemory memory      = VK_NULL_HANDLE;
        VkImageView    view        = VK_NULL_HANDLE;
        VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    };
    std::vector<LayerImage> m_layerImages;

    // Passes / framebuffers
    VkRenderPass  m_noAttachmentRenderPass  = VK_NULL_HANDLE;
    VkRenderPass  m_layerRenderPass         = VK_NULL_HANDLE;
    VkFramebuffer m_noAttachmentFramebuffer = VK_NULL_HANDLE;

    VkSampler m_sampler = VK_NULL_HANDLE;

    // Descriptors (binding 0 head image, 1 nodes, 2 counter)
    VkDescriptorSetLayout m_ppllDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_ppllDescSet    = VK_NULL_HANDLE;

    VkPipelineLayout m_collectPipelineLayout    = VK_NULL_HANDLE;
    VkPipelineLayout m_fullscreenPipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_collectPipeline = VK_NULL_HANDLE;
    VkPipeline m_sortPipeline    = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;

    // Readback staging
    VkBuffer       m_stagingBuffer     = VK_NULL_HANDLE;
    VkDeviceMemory m_stagingMemory     = VK_NULL_HANDLE;
    VkDeviceSize   m_stagingBufferSize = 0;
    void*          m_stagingMapped     = nullptr;

    // One-shot capture
    VkCommandPool   m_captureCmdPool = VK_NULL_HANDLE;
    VkCommandBuffer m_captureCmd     = VK_NULL_HANDLE;
    VkFence         m_captureFence   = VK_NULL_HANDLE;
};