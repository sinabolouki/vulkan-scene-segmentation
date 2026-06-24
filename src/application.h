#pragma once

#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "depth_peeling_renderer.h"
#include "scene.h"
#include "push_constants.h"
#include "mesh.h"

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
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;

    VulkanContext m_context;
    SwapchainManager m_swapchain;
    DepthPeelingRenderer m_depthPeeler;
    Scene m_scene;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
    uint32_t m_currentFrame = 0;

    // GPU Buffers
    VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer m_indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indexBufferMemory = VK_NULL_HANDLE;

    bool createCommandPool();
    bool createCommandBuffers();
    bool uploadSceneToGPU();
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};