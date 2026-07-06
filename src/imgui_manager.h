#pragma once

#include <vector>
#include <vulkan/vulkan.h>

struct GLFWwindow;
struct VulkanContext;
class SwapchainManager;

class ImGuiManager {
public:
    bool init(VulkanContext& context, SwapchainManager& swapchain, GLFWwindow* window);
    void destroy(VkDevice device);
    void recreate(VulkanContext& context, SwapchainManager& swapchain);

    // Call at the start of each frame (before any ImGui calls)
    void beginFrame();

    // Call after all ImGui::... calls, records draw commands into cmd
    void render(VkCommandBuffer cmd, VkExtent2D extent, VkFramebuffer framebuffer);

    VkRenderPass getRenderPass() const { return m_renderPass; }

    // Framebuffers for ImGui (one per swapchain image)
    VkFramebuffer getFramebuffer(uint32_t index) const { return m_framebuffers[index]; }

private:
    bool createRenderPass(VkDevice device, VkFormat swapchainFormat);
    bool createFramebuffers(VkDevice device, SwapchainManager& swapchain);
    void destroyFramebuffers(VkDevice device);

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkRenderPass     m_renderPass     = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;
};