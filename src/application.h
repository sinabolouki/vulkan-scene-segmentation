#pragma once

// Must come BEFORE any GLM include (scene.h / mesh.h include GLM).
// Even better: set it in CMake: target_compile_definitions(<target> PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE)
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

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
#elif defined(RENDER_METHOD_SAM2)
#include "sam2/sam2_renderer.h"
#include "sam2/sam2_job.h"      // includes sam2_client.h
#include "sam2/sam2_eval.h"
#include <future>
#include <memory>
#else
#error "No renderer selected. Configure using -DTHESIS_RENDERER=<name>."
#endif

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>
#include <chrono>

#include "imgui_manager.h"

class Application
{
    struct Camera
    {
        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 target   = glm::vec3(0.0f);
        glm::vec3 up       = glm::vec3(0.0f, 1.0f, 0.0f);

        // Orbit parameters
        float yaw      = 0.0f;  // horizontal angle (radians)
        float pitch    = 0.2f;  // vertical angle (radians)
        float distance = 5.0f;  // distance from target

        // Input state
        bool   leftMouseDown   = false;
        bool   middleMouseDown = false;
        double lastMouseX = 0, lastMouseY = 0;

        // Sensitivity
        float orbitSpeed = 0.005f;
        float panSpeed   = 0.005f;
        float zoomSpeed  = 0.1f;

        void updateFromOrbit()
        {
            float x = distance * cos(pitch) * cos(yaw);
            float y = distance * sin(pitch);
            float z = distance * cos(pitch) * sin(yaw);
            position = target + glm::vec3(x, y, z);
        }

        glm::mat4 getViewMatrix() const
        {
            return glm::lookAt(position, target, up);
        }
    };

public:
    bool init(uint32_t width, uint32_t height, const std::string &title,
              const std::string &scenePath);
    void run();
    void cleanup();

private:
    // Window
    GLFWwindow *m_window = nullptr;
    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;

    // Vulkan Core
    VulkanContext    m_context;
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
#elif defined(RENDER_METHOD_SAM2)
    Sam2Renderer m_renderer;
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
    VkBuffer       m_vertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer       m_indexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory m_indexBufferMemory  = VK_NULL_HANDLE;

    Camera m_camera;
    std::string m_cameraStatePath;
    bool m_cameraInitialized = false;

    // One-shot scene capture. The GPU copy is recorded before the ImGui pass.
    bool m_screenshotRequested = false;
    std::string m_screenshotSceneName = "scene";
    std::string m_screenshotStatus;
    VkBuffer m_screenshotBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_screenshotMemory = VK_NULL_HANDLE;
    VkDeviceSize m_screenshotSize = 0;
    VkExtent2D m_screenshotExtent = {0, 0};
    VkFormat m_screenshotFormat = VK_FORMAT_UNDEFINED;
    bool m_screenshotMemoryCoherent = false;

    // Init
    bool createCommandPool();
    bool createCommandBuffers();
    bool uploadSceneToGPU();
    bool loadTextures();

    // Frame
    void drawFrame();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void recreateSwapchain();   // NEW: single place for resize handling

    // Screenshots
    void drawScreenshotPanel();
    bool prepareScreenshot();
    void recordScreenshot(VkCommandBuffer cmd, uint32_t imageIndex);
    void finishScreenshot(VkFence submittedFrameFence);
    void destroyScreenshotBuffer();

    // Callbacks
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    // Helpers
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &memory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void initCamera();
    bool loadCameraState();
    bool saveCameraState() const;
    void updateCamera();
    void computeCameraMatrices(glm::mat4& view, glm::mat4& proj);
    double m_lastMouseX = 0, m_lastMouseY = 0;
    bool   m_firstMouse = true;

#ifdef RENDER_METHOD_SAM2
    // ─── SAM2 offline pipeline (m_renderer IS the Sam2Renderer) ───
    // Deferred action: capture needs Vulkan -> runs at the start of drawFrame
    bool m_sam2CaptureNow  = false;
    bool m_sam2AutoSegment = true;

    // Settings
    std::string m_sam2OutputDir      = "captures";
    std::string m_sam2Host           = "127.0.0.1";
    int         m_sam2Port           = 8765;
    int         m_sam2PointsPerSide  = 0;      // 0 = server default
    bool        m_sam2SaveMaskImages = true;

    // Data
    std::shared_ptr<const sam2::LayerSet> m_sam2LastCapture;   // shared with the worker
    std::vector<sam2::SegmentResult>      m_sam2Results;       // index == layer index

    // Async state
    sam2::Sam2Job m_sam2Job;
    bool          m_sam2JobActive = false;   // main-thread view: job started, not yet reaped
    std::string   m_sam2Status;

    std::future<std::pair<bool, std::string>> m_sam2HealthFuture;
    bool m_sam2HealthPending = false;

    // SAM2 vs union-find evaluation
    int              m_evalMinRegionPixels = 50;
    sam2::Evaluation m_evaluation;
    bool             m_hasEvaluation = false;

    void runSam2Capture();
    bool startSam2Segmentation();
    void startSam2HealthCheck();
    void pollSam2Job();
    void runSam2Evaluation();
    void drawSam2Panel();
    void drawEvaluationTable();
#endif
};
