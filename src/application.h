#pragma once

// CHOOSE RENDERING METHOD (uncomment ONE):
// #define RENDER_METHOD_PPLL              // Per-Pixel Linked List
#define RENDER_METHOD_DEPTH_PEELING   // Depth Peeling

#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "scene.h"
#include "push_constants.h"
#include "mesh.h"

#ifdef RENDER_METHOD_DEPTH_PEELING
    #include "depth_peeling_renderer.h"
#elif defined(RENDER_METHOD_PPLL)
    #include "ppll_renderer.h"
#else
    #error "No render method defined! Uncomment one in application.h"
#endif

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
    //Window
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;

    //Vulkan Core
    VulkanContext m_context;
    SwapchainManager m_swapchain;

    //Renderer (only one is compiled)
#ifdef RENDER_METHOD_DEPTH_PEELING
    DepthPeelingRenderer m_renderer;
#elif defined(RENDER_METHOD_PPLL)
    PPLLRenderer m_renderer;
#endif

    // Scene
    Scene m_scene;

    // Commands
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
    uint32_t m_currentFrame = 0;

    // GPU Buffers
    VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer m_indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_indexBufferMemory = VK_NULL_HANDLE;

    //Init
    bool createCommandPool();
    bool createCommandBuffers();
    bool uploadSceneToGPU();

    //Frame
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    //Helpers
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};