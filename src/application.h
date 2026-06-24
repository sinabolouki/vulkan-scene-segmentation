#pragma once

#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "pipeline_builder.h"
#include "shader_utils.h"
#include "mesh.h"
#include "push_constants.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

class Application {
public:
    bool init(uint32_t width, uint32_t height, const std::string& title);
    void run();
    void cleanup();

private:
    // ─── Window ───
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;

    // ─── Vulkan Core ───
    VulkanContext m_context;
    SwapchainManager m_swapchain;

    // ─── Commands ───
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
    uint32_t m_currentFrame = 0;

    // ─── Pipeline ───
    VkPipelineLayout m_scenePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_initPipeline = VK_NULL_HANDLE;
    // Later: VkPipeline m_peelPipeline, m_compositePipeline

    // ─── Scene Geometry ───
    VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer m_indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indexBufferMemory = VK_NULL_HANDLE;
    uint32_t m_indexCount = 0;

    // ─── Init Functions ───
    bool createCommandPool();
    bool createCommandBuffers();
    bool createPipelineLayout();
    bool createPipelines();
    bool createSceneGeometry();

    // ─── Frame ───
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    // ─── Helpers ───
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};