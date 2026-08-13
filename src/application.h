#pragma once

// CHOOSE RENDERING METHOD (uncomment ONE):
// #define RENDER_METHOD_PPLL              // Per-Pixel Linked List
// #define RENDER_METHOD_DEPTH_PEELING     // Depth Peeling
#define RENDER_METHOD_UNION_FIND        // Union-Find PPLL
// #define RENDER_METHOD_SIMPLE              // Simple forward (for testing)

#include "vulkan_context.h"
#include "swapchain_manager.h"
#include "scene.h"
#include "push_constants.h"
#include "mesh.h"
#include "texture_manager.h"

#ifdef RENDER_METHOD_DEPTH_PEELING
#include "depth_peeling_renderer.h"
#elif defined(RENDER_METHOD_PPLL)
#include "ppll_renderer.h"
#elif defined(RENDER_METHOD_UNION_FIND)
#include "uf_renderer.h"
#elif defined(RENDER_METHOD_SIMPLE)
#include "simple_renderer.h"
#else
#error "No render method defined! Uncomment one in application.h"
#endif

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

#include "imgui_manager.h"
#include <chrono>

class Application
{
    struct Camera
    {
        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 target = glm::vec3(0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

        // Orbit parameters
        float yaw = 0.0f;      // horizontal angle (radians)
        float pitch = 0.2f;    // vertical angle (radians)
        float distance = 5.0f; // distance from target

        // Input state
        bool leftMouseDown = false;
        bool middleMouseDown = false;
        double lastMouseX = 0, lastMouseY = 0;

        // Sensitivity
        float orbitSpeed = 0.005f;
        float panSpeed = 0.005f;
        float zoomSpeed = 0.1f;

        void updateFromOrbit()
        {
            float x = distance * cos(pitch) * cos(yaw);
            float y = distance * sin(pitch);
            float z = distance * cos(pitch) * sin(yaw);
            position = target + glm::vec3(x, y, z);
        }

        glm::mat4 getViewMatrix() const
        {
            glm::mat4 view = glm::lookAt(position, target, up);
            return view;
        }
    };

public:
    bool init(uint32_t width, uint32_t height, const std::string &title);
    void run();
    void cleanup();

private:
    // Window
    GLFWwindow *m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;

    // Vulkan Core
    VulkanContext m_context;
    SwapchainManager m_swapchain;

    // Renderer (only one is compiled)
#ifdef RENDER_METHOD_DEPTH_PEELING
    DepthPeelingRenderer m_renderer;
#elif defined(RENDER_METHOD_PPLL)
    PPLLRenderer m_renderer;
#elif defined(RENDER_METHOD_UNION_FIND)
    UFRenderer m_renderer;
#elif defined(RENDER_METHOD_SIMPLE)
    SimpleRenderer m_renderer;
#endif

    // Texture Manager (shared across renderers)
    TextureManager m_textureManager;

    ImGuiManager m_imguiManager;

    // Frame timing
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    double m_frameTimeMs = 0.0;
    double m_fps = 0.0;

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

    Camera m_camera;

    // Init
    bool createCommandPool();
    bool createCommandBuffers();
    bool uploadSceneToGPU();
    bool loadTextures();

    // Frame
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    // Callbacks
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

    // Helpers
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    void initCamera();
    void updateCamera();
    double m_lastMouseX = 0, m_lastMouseY = 0;
    bool   m_firstMouse = true;
};