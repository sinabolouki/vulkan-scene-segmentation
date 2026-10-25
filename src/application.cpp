#include "application.h"

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <array>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <cmath>
#include <system_error>
#include <imgui.h>
#ifdef RENDER_METHOD_SAM2
#include "sam2/sam2_io.h"
#endif

void Application::framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
    auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
    app->m_framebufferResized = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

bool Application::init(uint32_t width, uint32_t height, const std::string &title,
                       const std::string &scenePath)
{
    m_width = width;
    m_height = height;
    m_cameraInitialized = false;
    m_cameraStatePath.clear();

    // GLFW
    if (!glfwInit())
        return false;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window)
        return false;
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);

    // Vulkan
    if (!m_context.init(m_window))
        return false;
    if (!m_swapchain.init(m_context, width, height))
        return false;
    if (!createCommandPool())
        return false;
    if (!createCommandBuffers())
        return false;

    // Scene
    if (!m_scene.loadFromOBJ(scenePath))
    {
        std::cerr << "Failed to load scene: " << scenePath << std::endl;
        return false;
    }

    // One camera file per OBJ; an absolute path also survives working-directory changes.
    std::error_code cameraPathError;
    const auto absoluteScenePath = std::filesystem::absolute(scenePath, cameraPathError);
    if (!cameraPathError)
        m_cameraStatePath = absoluteScenePath.lexically_normal().string() + ".camera";
    else
        std::cerr << "[Camera] Could not resolve camera file path: "
                  << cameraPathError.message() << '\n';

    if (!uploadSceneToGPU())
        return false;

    glfwSetKeyCallback(m_window, keyCallback);

    // Textures (must happen BEFORE renderer init)
    if (!loadTextures())
        return false;

    // Give texture manager to renderer
    m_renderer.setTextureManager(&m_textureManager);

    // Renderer
#ifdef RENDER_METHOD_SAM2
    {
        Sam2Renderer::Config cfg{};
        cfg.width = m_swapchain.getExtent().width;
        cfg.height = m_swapchain.getExtent().height;
        cfg.maxLayers = UFRenderer::NUM_LAYERS;
        if (!m_renderer.init(m_context, cfg))
            return false;
    }
#else
    if (!m_renderer.init(m_context, m_swapchain))
        return false;
#endif

#ifdef RENDER_METHOD_SAM2
    m_sam2Status = "Checking SAM2 server...";
    startSam2HealthCheck(); // async: app starts immediately, result appears in the panel
#endif

    if (!m_imguiManager.init(m_context, m_swapchain, m_window))
        return false;
    m_lastFrameTime = std::chrono::high_resolution_clock::now();

#ifdef RENDER_METHOD_DEPTH_PEELING
    std::cout << "[App] Render method: DEPTH PEELING" << std::endl;
#elif defined(RENDER_METHOD_PPLL)
    std::cout << "[App] Render method: PER-PIXEL LINKED LIST" << std::endl;
#endif

    // Initialize camera after scene is loaded
    initCamera();

    std::cout << "[App] Initialization complete." << std::endl;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Record Command Buffer
// ─────────────────────────────────────────────────────────────────────────────

void Application::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Camera
    const VkExtent2D extent = m_swapchain.getExtent();
    glm::mat4 view, proj;
    computeCameraMatrices(view, proj);
    glm::mat4 viewProj = proj * view;

#ifdef RENDER_METHOD_SAM2
    // 1) PPLL collect + sort + resolve layer 0 into the renderer's offscreen image
    m_renderer.recordPreview(cmd,
                             m_vertexBuffer, m_indexBuffer,
                             m_scene.getObjects(),
                             viewProj, view, proj);

    // 2) Blit layer 0 into the swapchain image and leave it in PRESENT_SRC_KHR,
    //    which is exactly the initialLayout ImGui's LOAD_OP_LOAD render pass expects.
    m_renderer.recordBlitPreviewToSwapchain(cmd,
                                            m_swapchain.getImage(imageIndex),
                                            extent);
#else
    m_renderer.recordCommands(cmd, m_context, m_swapchain,
                              m_vertexBuffer, m_indexBuffer,
                              m_scene.getObjects(),
                              viewProj, view, proj, imageIndex);
#endif

    // 3) ImGui on top
    m_imguiManager.render(cmd, extent, m_imguiManager.getFramebuffer(imageIndex));

    vkEndCommandBuffer(cmd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw Frame
// ─────────────────────────────────────────────────────────────────────────────

void Application::drawFrame()
{
    // Frame timing
    auto now = std::chrono::high_resolution_clock::now();
    m_frameTimeMs = std::chrono::duration<double, std::milli>(now - m_lastFrameTime).count();
    m_fps = 1000.0 / m_frameTimeMs;
    m_lastFrameTime = now;

#ifdef RENDER_METHOD_SAM2
    // Collect finished results / health check (cheap, non-blocking)
    pollSam2Job();

    // Capture needs the GPU -> done here, outside the frame's command buffer
    if (m_sam2CaptureNow)
    {
        m_sam2CaptureNow = false;
        runSam2Capture();
        if (m_sam2AutoSegment && m_sam2LastCapture)
            startSam2Segmentation(); // returns immediately
    }
#endif

    // Begin ImGui frame (before fence wait is fine — ImGui just builds draw lists)
    m_imguiManager.beginFrame();
    updateCamera();

    // ─── ImGui UI ───
#ifdef RENDER_METHOD_UNION_FIND
    ImGui::Begin("Union-Find Renderer");

    ImGui::Separator();
    const char *mergeModes[] = {
        "Balanced (weighted sum)",
        "Color Priority (color veto → spatial)",
        "Space Priority (spatial veto → color)"};
    int currentMode = (int)m_renderer.mergeMode;
    if (ImGui::Combo("Merge Mode", &currentMode, mergeModes, IM_ARRAYSIZE(mergeModes)))
    {
        m_renderer.mergeMode = (uint32_t)currentMode;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "Mode 0: All terms weighted equally into one cost.\n"
            "Mode 1: Color checked first — if colors differ, fragments\n"
            "        are never merged regardless of spatial proximity.\n"
            "Mode 2: Spatial checked first — if geometry says different,\n"
            "        fragments are never merged regardless of color.");
    }

    // ─── Connectivity Parameters ───
    ImGui::Text("Connectivity Parameters");
    ImGui::Separator();
    ImGui::SliderFloat("Plane Threshold", &m_renderer.planeThreshold, 0.1f, 10.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Tangent-plane residual tolerance (x pixelStep).\n"
                          "Higher = more permissive on curved surfaces.\n"
                          "Lower = stricter, may split curved objects.");

    ImGui::SliderFloat("Proximity Scale", &m_renderer.proximityScale, 0.5f, 10.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Proximity fallback for sharp edges/creases (x pixelStep).\n"
                          "Higher = merges cube edges more aggressively.\n"
                          "Lower = may split objects at sharp creases.");

    ImGui::SliderFloat("Color Threshold", &m_renderer.colorThreshold, 0.05f, 1.5f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Chromaticity distance veto.\n"
                          "Lower = stricter (splits at subtle color changes).\n"
                          "Higher = more permissive (only rejects very different materials).\n"
                          "0.4 = good default for textured scenes.");

    ImGui::Separator();

    // ─── Performance ───
    ImGui::Text("FPS: %.1f (%.2f ms)", m_fps, m_frameTimeMs);
    ImGui::Text("Camera data: ");
    ImGui::Text("pos  X: %.2f, Y: %.2f, Z: %.2f", m_camera.position.x, m_camera.position.y, m_camera.position.z);
    ImGui::Text("target  X: %.2f, Y: %.2f, Z: %.2f", m_camera.target.x, m_camera.target.y, m_camera.target.z);
    ImGui::Text("Near/Far: %.3f / %.1f", m_renderer.nearPlane, m_renderer.farPlane);
    ImGui::Text("Camera distance: %.2f", m_camera.distance);
    ImGui::Text("Camera Pitch: %.2f, Yaw: %.2f", m_camera.pitch, m_camera.yaw);
    ImGui::Separator();

    if (m_renderer.enableTiming)
    {
        auto &t = m_renderer.timingResults;
        ImGui::Text("GPU Timing (ms):");
        ImGui::Separator();
        ImGui::Text("  Clear:        %6.3f", t.clear);
        ImGui::Text("  Collect:      %6.3f", t.collect);
        ImGui::Text("  Sort:         %6.3f", t.sort);
        ImGui::Text("  Init Parents: %6.3f", t.initParents);
        ImGui::Text("  Merge:        %6.3f", t.merge);
        ImGui::Text("  Compress:     %6.3f", t.compress);
        ImGui::Separator();

        for (uint32_t i = 0; i < m_renderer.NUM_LAYERS; i++)
        {
            if (ImGui::TreeNode((void *)(intptr_t)i, "Peel %d: %6.3f ms", i, t.peels[i].total))
            {
                ImGui::Text("  Clear Table: %6.3f", t.peels[i].clearTable);
                ImGui::Text("  Register:    %6.3f", t.peels[i].registerRoots);
                ImGui::Text("  Resolve:     %6.3f", t.peels[i].resolve);
                ImGui::TreePop();
            }
        }
        ImGui::Text("  Peels Total:  %6.3f", t.peelTotal);
        ImGui::Separator();
        ImGui::Text("  Composite:    %6.3f", t.composite);
        ImGui::Text("  ════════════════════");
        ImGui::Text("  TOTAL:        %6.3f", t.total);
    }
    ImGui::Checkbox("Enable GPU Timing", &m_renderer.enableTiming);

    ImGui::Separator();

    const char *layerNames[] = {
        "Composite",
        "Layer 0", "Layer 1", "Layer 2", "Layer 3",
        "Layer 4", "Layer 5", "Layer 6", "Layer 7",
        "Layer 8", "Layer 9", "Layer 10", "Layer 11",
        "Layer 12", "Layer 13", "Layer 14", "Layer 15"};

    int layerIdx = m_renderer.debugDisplayLayer + 1;
    ImGui::Combo(
        "Display",
        &layerIdx,
        layerNames,
        IM_ARRAYSIZE(layerNames));
    m_renderer.debugDisplayLayer = layerIdx - 1;
    
    ImGui::Separator();
    ImGui::Checkbox("Bypass Union-Find (show front only)", &m_renderer.bypassUnionFind);

    // In your ImGui panel:
    ImGui::Separator();
    ImGui::Text("Debug Visualization");

    const char *debugModes[] = {
        "Normal Rendering",    // -1
        "Region ID (root)",    //  0
        "Object ID",           //  1
        "Depth",               //  2
        "View Normals",        //  3
        "Fragment Color",      //  4
        "Fragment Count Heat", //  5
        "Peel Layer",          //  6
        "Region Rank Span"     //  7
    };

    // Offset by 1 so index 0 = "Normal Rendering" = debugVisualizationMode -1
    int current = m_renderer.debugVisualizationMode + 1;
    if (ImGui::Combo("Vis Mode", &current, debugModes, IM_ARRAYSIZE(debugModes)))
    {
        m_renderer.debugVisualizationMode = current - 1;
    }

    ImGui::Separator();
    ImGui::Text("GPU Debug Info (Peel Layer 0):");
    ImGui::Text("  Min maxPosition:   %u", m_renderer.debugGPUInfo.minMaxPos);
    ImGui::Text("  Registered roots:  %u / %u", m_renderer.debugGPUInfo.numRegisteredRoots, UFRenderer::MAX_GLOBAL_ROOTS);

    if (m_renderer.debugGPUInfo.minMaxPos > 0)
    {
        ImGui::TextColored(ImVec4(1, 1, 0, 1),
                           "  Warning: No root has maxPos=0 (min=%u)", m_renderer.debugGPUInfo.minMaxPos);
    }
    if (m_renderer.debugGPUInfo.numRegisteredRoots >= UFRenderer::MAX_GLOBAL_ROOTS)
    {
        ImGui::TextColored(ImVec4(1, 0, 0, 1),
                           "  ERROR: Root table full! Increase MAX_GLOBAL_ROOTS");
    }

    // ─── Debug Info ───
    if (ImGui::CollapsingHeader("Debug Info"))
    {
        ImGui::Text("Resolution: %u x %u", m_renderer.getWidth(), m_renderer.getHeight());
        ImGui::Text("Node Capacity: %u", m_renderer.getNodeBufferCapacity());
        ImGui::Text("Node Size: %zu bytes", sizeof(UFFragmentNode));
        ImGui::Text("Node Buffer: %.1f MB",
                    m_renderer.getNodeBufferCapacity() * sizeof(UFFragmentNode) / (1024.0f * 1024.0f));
        ImGui::Text("Near/Far: %.3f / %.1f", m_renderer.nearPlane, m_renderer.farPlane);
    }

    ImGui::End();
#elif defined(RENDER_METHOD_PPLL)
    ImGui::Begin("PPLL Renderer");
    ImGui::Text("FPS: %.1f (%.2f ms)", m_fps, m_frameTimeMs);
    ImGui::Separator();
    // ImGui::SliderFloat("Depth Threshold", &m_renderer.depthThreshold, 0.001f, 0.1f, "%.4f");
    // ImGui::SliderFloat("Color Threshold", &m_renderer.colorThreshold, 0.01f, 0.5f, "%.3f");
    int grow = (int)m_renderer.growIterations;
    ImGui::SliderInt("Grow Iterations", &grow, 1, 100);
    m_renderer.growIterations = (uint32_t)grow;
    ImGui::Separator();
    const char *layerNames[] = {"Composite", "Layer 0", "Layer 1", "Layer 2", "Layer 3"};
    int layerIdx = m_renderer.debugDisplayLayer + 1; // -1 → 0, 0→1, etc.
    ImGui::Combo("Display", &layerIdx, layerNames, 5);
    m_renderer.debugDisplayLayer = layerIdx - 1;
    ImGui::End();
#elif defined(RENDER_METHOD_SIMPLE)
    ImGui::Begin("Simple Renderer");
    ImGui::Text("FPS: %.1f (%.2f ms)", m_fps, m_frameTimeMs);
    ImGui::Separator();
    ImGui::Text("Scene: %s", "sponza");
    ImGui::Text("Objects: %u", (uint32_t)m_scene.getObjects().size());
    ImGui::Text("Vertices: %u", m_scene.getTotalVertexCount());
    ImGui::Text("Triangles: %u", m_scene.getTotalIndexCount() / 3);
    ImGui::Text("Textures: %u", m_textureManager.getTextureCount());
    ImGui::Separator();
    ImGui::Text("Camera distance: %.2f", m_camera.distance);
    ImGui::Text("Near/Far: %.3f / %.1f", m_renderer.nearPlane, m_renderer.farPlane);
    ImGui::Text("CameraPos: %.3f , %.3f , %.3f", m_camera.position.x, m_camera.position.y, m_camera.position.z);
    ImGui::Text("CameraTarget: %.3f , %.3f , %.3f", m_camera.target.x, m_camera.target.y, m_camera.target.z);
    ImGui::End();
#elif defined(RENDER_METHOD_SAM2)
    drawSam2Panel();
#endif
    // ─── End ImGui UI ───

    VkFence inFlightFence = m_swapchain.getInFlightFence(m_currentFrame);
    vkWaitForFences(m_context.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        m_context.device, m_swapchain.getSwapchain(), UINT64_MAX,
        m_swapchain.getImageAvailableSemaphore(m_currentFrame),
        VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        ImGui::Render(); // close the ImGui frame we already started
        recreateSwapchain();
        return;
    }

    vkResetFences(m_context.device, 1, &inFlightFence);
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = {m_swapchain.getImageAvailableSemaphore(m_currentFrame)};
    // The SAM2 blit barrier uses COLOR_ATTACHMENT_OUTPUT as its srcStage, so it chains
    // correctly with this wait stage; no need to add TRANSFER here.
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signalSemaphores[] = {m_swapchain.getRenderFinishedSemaphore(m_currentFrame)};

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(m_context.graphics_queue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
    {
        std::cerr << "Failed to submit command buffer" << std::endl;
    }

    result = m_swapchain.present(m_context.present_queue, imageIndex, signalSemaphores[0]);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized)
    {
        m_framebufferResized = false;
        recreateSwapchain();
    }

    m_currentFrame = (m_currentFrame + 1) % SwapchainManager::MAX_FRAMES_IN_FLIGHT;
}

// Add the callback implementation:
void Application::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
#if !defined(RENDER_METHOD_SIMPLE) && !defined(RENDER_METHOD_SAM2)
    if (action != GLFW_PRESS)
        return;

    auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));

    // Inside keyCallback:

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        auto *app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        switch (key)
        {
        case GLFW_KEY_0:
            app->m_renderer.debugDisplayLayer = -1;
            break; // Normal
        case GLFW_KEY_1:
            app->m_renderer.debugDisplayLayer = 0;
            break; // Layer 0
        case GLFW_KEY_2:
            app->m_renderer.debugDisplayLayer = 1;
            break; // Layer 1
        case GLFW_KEY_3:
            app->m_renderer.debugDisplayLayer = 2;
            break; // Layer 2
        case GLFW_KEY_4:
            app->m_renderer.debugDisplayLayer = 3;
            break; // Layer 3
        }
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Loop & Cleanup
// ─────────────────────────────────────────────────────────────────────────────

void Application::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(m_context.device);
}

void Application::cleanup()
{
    // Save before GPU/window teardown. Failed initialization must not overwrite a pose.
    saveCameraState();
    m_cameraInitialized = false;

    VkDevice device = m_context.device;
#ifdef RENDER_METHOD_SAM2
    // The worker touches no Vulkan objects, but it must stop before members are destroyed.
    m_sam2Job.cancel();
    m_sam2Job.join(); // waits for at most the current SAM2 request
    if (m_sam2HealthFuture.valid())
        m_sam2HealthFuture.wait();
#endif
    vkDeviceWaitIdle(device);

    m_imguiManager.destroy(device);
    m_renderer.destroy(device); // first: its pipeline layout uses the texture set layout
    m_textureManager.destroy(device);

    vkDestroyBuffer(device, m_indexBuffer, nullptr);
    vkFreeMemory(device, m_indexBufferMemory, nullptr);
    vkDestroyBuffer(device, m_vertexBuffer, nullptr);
    vkFreeMemory(device, m_vertexBufferMemory, nullptr);

    vkDestroyCommandPool(device, m_commandPool, nullptr);
    m_swapchain.destroy(device);
    m_context.destroy();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene Upload & Helpers (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

bool Application::uploadSceneToGPU()
{
    const auto &vertices = m_scene.getVertices();
    const auto &indices = m_scene.getIndices();

    VkDeviceSize vertexSize = sizeof(Vertex) * vertices.size();
    if (!createBuffer(vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_vertexBuffer, m_vertexBufferMemory))
        return false;
    void *data;
    vkMapMemory(m_context.device, m_vertexBufferMemory, 0, vertexSize, 0, &data);
    memcpy(data, vertices.data(), vertexSize);
    vkUnmapMemory(m_context.device, m_vertexBufferMemory);

    VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
    if (!createBuffer(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_indexBuffer, m_indexBufferMemory))
        return false;
    vkMapMemory(m_context.device, m_indexBufferMemory, 0, indexSize, 0, &data);
    memcpy(data, indices.data(), indexSize);
    vkUnmapMemory(m_context.device, m_indexBufferMemory);

    return true;
}

bool Application::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_context.graphics_queue_family;
    return vkCreateCommandPool(m_context.device, &poolInfo, nullptr, &m_commandPool) == VK_SUCCESS;
}

bool Application::createCommandBuffers()
{
    m_commandBuffers.resize(SwapchainManager::MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
    return vkAllocateCommandBuffers(m_context.device, &allocInfo, m_commandBuffers.data()) == VK_SUCCESS;
}

bool Application::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags properties,
                               VkBuffer &buffer, VkDeviceMemory &memory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_context.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(m_context.device, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);
    if (vkAllocateMemory(m_context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        return false;

    vkBindBufferMemory(m_context.device, buffer, memory, 0);
    return true;
}

uint32_t Application::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_context.physical_device, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

void Application::initCamera()
{
    glm::vec3 center = m_scene.getCenter();
    float radius = m_scene.getRadius();

    m_camera.target = center;
    m_camera.distance = glm::max(radius, 0.01f);
    m_camera.yaw = 0.0f;
    m_camera.pitch = 0.3f;
    m_camera.updateFromOrbit();

    // Keep the scene-based default if no valid saved pose exists.
    loadCameraState();
    m_cameraInitialized = true;
}

bool Application::loadCameraState()
{
    if (m_cameraStatePath.empty())
        return false;

    std::ifstream file(m_cameraStatePath);
    if (!file)
        return false; // Normal on the first run for this scene.
    file.imbue(std::locale::classic());

    // Parse into a copy so a malformed file cannot partially change the camera.
    Camera saved = m_camera;
    std::string magic;
    int version = 0;
    if (!(file >> magic >> version) || magic != "THESIS_CAMERA" || version != 1 ||
        !(file >> saved.target.x >> saved.target.y >> saved.target.z
               >> saved.yaw >> saved.pitch >> saved.distance))
    {
        std::cerr << "[Camera] Ignoring invalid camera file: " << m_cameraStatePath << '\n';
        return false;
    }

    if (!std::isfinite(saved.target.x) || !std::isfinite(saved.target.y) ||
        !std::isfinite(saved.target.z) || !std::isfinite(saved.yaw) ||
        !std::isfinite(saved.pitch) || !std::isfinite(saved.distance) ||
        saved.distance < 0.01f || saved.pitch < -1.5f || saved.pitch > 1.5f)
    {
        std::cerr << "[Camera] Ignoring out-of-range camera state: " << m_cameraStatePath << '\n';
        return false;
    }

    saved.updateFromOrbit();
    if (!std::isfinite(saved.position.x) || !std::isfinite(saved.position.y) ||
        !std::isfinite(saved.position.z))
    {
        std::cerr << "[Camera] Ignoring invalid camera position: " << m_cameraStatePath << '\n';
        return false;
    }

    m_camera = saved;
    std::cout << "[Camera] Restored pose from " << m_cameraStatePath << '\n';
    return true;
}

bool Application::saveCameraState() const
{
    if (!m_cameraInitialized || m_cameraStatePath.empty())
        return false;

    std::ofstream file(m_cameraStatePath, std::ios::trunc);
    if (!file)
    {
        std::cerr << "[Camera] Could not write camera file: " << m_cameraStatePath << '\n';
        return false;
    }
    file.imbue(std::locale::classic());
    file << std::setprecision(std::numeric_limits<float>::max_digits10)
         << "THESIS_CAMERA 1\n"
         << m_camera.target.x << ' ' << m_camera.target.y << ' ' << m_camera.target.z << '\n'
         << m_camera.yaw << ' ' << m_camera.pitch << ' ' << m_camera.distance << '\n';
    file.close();
    if (!file)
    {
        std::cerr << "[Camera] Failed to save camera file: " << m_cameraStatePath << '\n';
        return false;
    }

    std::cout << "[Camera] Saved pose to " << m_cameraStatePath << '\n';
    return true;
}

void Application::updateCamera()
{
    // Don't process camera input if ImGui wants the mouse
    if (ImGui::GetIO().WantCaptureMouse)
    {
        m_firstMouse = true; // reset so we don't get a jump when returning to viewport
        return;
    }

    double mouseX, mouseY;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    if (m_firstMouse)
    {
        m_lastMouseX = mouseX;
        m_lastMouseY = mouseY;
        m_firstMouse = false;
    }

    double dx = mouseX - m_lastMouseX;
    double dy = mouseY - m_lastMouseY;
    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;

    // Left mouse: orbit
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        m_camera.yaw -= float(dx) * m_camera.orbitSpeed;
        m_camera.pitch += float(dy) * m_camera.orbitSpeed;
        m_camera.pitch = glm::clamp(m_camera.pitch, -1.5f, 1.5f);
        m_camera.updateFromOrbit();
    }

    // Middle mouse: pan
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        glm::vec3 forward = glm::normalize(m_camera.target - m_camera.position);
        glm::vec3 right = glm::normalize(glm::cross(forward, m_camera.up));
        glm::vec3 camUp = glm::normalize(glm::cross(right, forward));

        float panScale = m_camera.distance * m_camera.panSpeed;
        m_camera.target -= right * float(dx) * panScale;
        m_camera.target += camUp * float(dy) * panScale;
        m_camera.updateFromOrbit();
    }

    // Scroll: zoom (use ImGui's scroll data since ImGui still handles the callback)
    // Alternative: read scroll via ImGui IO
    float scroll = ImGui::GetIO().MouseWheel;
    if (scroll != 0.0f)
    {
        m_camera.distance -= scroll * m_camera.distance * m_camera.zoomSpeed;
        m_camera.distance = glm::max(m_camera.distance, 0.01f);
        m_camera.updateFromOrbit();
    }
}

void Application::computeCameraMatrices(glm::mat4 &view, glm::mat4 &proj)
{
    const VkExtent2D extent = m_swapchain.getExtent();
    const float aspect = (float)extent.width / (float)extent.height;

    m_renderer.nearPlane = 0.01f;
    m_renderer.farPlane = m_camera.distance * 10.0f;

    view = m_camera.getViewMatrix();
    proj = glm::perspective(glm::radians(45.0f), aspect,
                            m_renderer.nearPlane, m_renderer.farPlane);
}

bool Application::loadTextures()
{
    if (!m_textureManager.init(m_context))
    {
        std::cerr << "[App] Failed to init texture manager" << std::endl;
        return false;
    }

    auto texturePaths = m_scene.getTextureFilenames();
    std::cout << "[App] Found " << texturePaths.size() << " textures to load" << std::endl;

    if (!texturePaths.empty())
    {
        if (!m_textureManager.loadTextures(m_context, texturePaths))
        {
            std::cerr << "[App] Failed to load textures" << std::endl;
            return false;
        }
        m_scene.resolveTextureIndices(m_textureManager.getLoadedPaths());
    }

    return true;
}

void Application::recreateSwapchain()
{
    // Minimized window: wait until it has a real size again (fix #5)
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while (w == 0 || h == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(m_window, &w, &h);
    }

    vkDeviceWaitIdle(m_context.device);

    m_swapchain.recreate(m_context, (uint32_t)w, (uint32_t)h);

#ifdef RENDER_METHOD_SAM2
    // Sam2Renderer has its own signature (fix #1)
    m_renderer.recreate(m_context,
                        m_swapchain.getExtent().width,
                        m_swapchain.getExtent().height);
#else
    m_renderer.recreate(m_context, m_swapchain);
#endif

    m_imguiManager.recreate(m_context, m_swapchain);
}
#ifdef RENDER_METHOD_SAM2
// ─────────────────────────────────────────────────────────────────────────────
// SAM2: capture (main thread, needs Vulkan)
// ─────────────────────────────────────────────────────────────────────────────

void Application::runSam2Capture()
{
    if (m_sam2Job.isRunning())
    {
        m_sam2Status = "Segmentation running: cancel it or wait before capturing.";
        return;
    }

    vkDeviceWaitIdle(m_context.device); // frames in flight may still use the buffers

    glm::mat4 view, proj;
    computeCameraMatrices(view, proj);
    const glm::mat4 viewProj = proj * view;

    try
    {
        auto layers = std::make_shared<sam2::LayerSet>(
            m_renderer.captureSnapshot(m_context, m_vertexBuffer, m_indexBuffer,
                                       m_scene.getObjects(), viewProj, view, proj));

        size_t nonEmpty = 0;
        for (const auto &l : *layers)
            if (!sam2::isLayerEmpty(l))
                ++nonEmpty;

        sam2::saveCapturedLayers(*layers, m_sam2OutputDir, /*skipEmpty=*/true);
        sam2::saveRegionLabelMaps(*layers, (std::filesystem::path(m_sam2OutputDir) / "eval").string());

        m_sam2LastCapture = std::move(layers); // shared_ptr<LayerSet> -> shared_ptr<const LayerSet>
        m_sam2Results.clear();
        m_hasEvaluation = false;

        m_sam2Status = "Captured " + std::to_string(m_sam2LastCapture->size()) + " layers (" + std::to_string(nonEmpty) + " non-empty).";
    }
    catch (const std::exception &e)
    {
        m_sam2LastCapture.reset();
        m_sam2Results.clear();
        m_sam2Status = std::string("Capture failed: ") + e.what();
        std::cerr << "[SAM2] " << m_sam2Status << std::endl;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SAM2: start async segmentation (non-blocking, no Vulkan -> safe from ImGui)
// ─────────────────────────────────────────────────────────────────────────────

bool Application::startSam2Segmentation()
{
    if (!m_sam2LastCapture || m_sam2LastCapture->empty())
    {
        m_sam2Status = "Nothing to segment: capture first.";
        return false;
    }
    if (m_sam2Job.isRunning())
        return false;

    sam2::JobConfig cfg;
    cfg.host = m_sam2Host;
    cfg.port = m_sam2Port;
    cfg.params.pointsPerSide = m_sam2PointsPerSide;
    cfg.maskDir = (std::filesystem::path(m_sam2OutputDir) / "masks").string();
    cfg.saveImages = m_sam2SaveMaskImages;

    // Pre-fill one "pending" row per layer (ok=false, error empty)
    const auto &layers = *m_sam2LastCapture;
    m_sam2Results.assign(layers.size(), sam2::SegmentResult{});
    for (size_t i = 0; i < layers.size(); ++i)
    {
        m_sam2Results[i].layer = int(i);
        m_sam2Results[i].width = layers[i].width;
        m_sam2Results[i].height = layers[i].height;
    }

    if (!m_sam2Job.start(m_sam2LastCapture, cfg))
    {
        m_sam2Status = "Could not start segmentation job.";
        return false;
    }
    m_sam2JobActive = true;
    m_hasEvaluation = false;
    m_sam2Status = "Segmenting...";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SAM2 vs union-find evaluation (main thread, CPU only)
// ─────────────────────────────────────────────────────────────────────────────

void Application::runSam2Evaluation()
{
    if (!m_sam2LastCapture || m_sam2Results.empty())
        return;

    m_evaluation = sam2::evaluateSamVsUnionFind(*m_sam2LastCapture, m_sam2Results,
                                                uint32_t(m_evalMinRegionPixels));
    m_hasEvaluation = true;

    if (m_evaluation.layers.empty())
        return; // keep the job's status message (failed / cancelled)

    const std::string csv = (std::filesystem::path(m_sam2OutputDir) / "eval" / "evaluation.csv").string();
    sam2::saveEvaluationCSV(m_evaluation, csv);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "Evaluated %zu layers: F1@0.5 = %.3f, UF covering = %.3f (%s)",
                  m_evaluation.layers.size(), m_evaluation.total.f1At50,
                  m_evaluation.total.ufCovering, csv.c_str());
    m_sam2Status = buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// SAM2: async health check
// ─────────────────────────────────────────────────────────────────────────────

void Application::startSam2HealthCheck()
{
    if (m_sam2HealthPending)
        return; // never overwrite an unfinished future (its dtor would block)

    const std::string host = m_sam2Host; // copies: the lambda must not touch members
    const int port = m_sam2Port;

    m_sam2HealthFuture = std::async(std::launch::async, [host, port]()
                                    {
        sam2::Sam2Client client(host, port);
        std::string info, err;
        const bool ok = client.health(info, err);
        return std::make_pair(ok, ok ? info : err); });
    m_sam2HealthPending = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SAM2: called once per frame on the main thread
// ─────────────────────────────────────────────────────────────────────────────

void Application::pollSam2Job()
{
    // 1) Read the running flag BEFORE taking results: the worker queues every result
    //    before it clears the flag, so "not running" + takeResults() == everything.
    const bool running = m_sam2Job.isRunning();

    for (auto &r : m_sam2Job.takeResults())
        if (r.layer >= 0 && size_t(r.layer) < m_sam2Results.size())
            m_sam2Results[size_t(r.layer)] = std::move(r);

    if (m_sam2JobActive && !running)
    {
        m_sam2JobActive = false;
        m_sam2Job.join(); // thread has exited (or is exiting) -> cheap
        m_sam2Status = m_sam2Job.progress().message;
        runSam2Evaluation();
    }

    // 2) Health check result
    if (m_sam2HealthPending && m_sam2HealthFuture.valid() &&
        m_sam2HealthFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
        auto [ok, msg] = m_sam2HealthFuture.get();
        m_sam2HealthPending = false;
        m_sam2Status = ok ? ("Server OK: " + msg) : ("Server not reachable: " + msg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SAM2: ImGui panel
// ─────────────────────────────────────────────────────────────────────────────

void Application::drawSam2Panel()
{
    const bool running = m_sam2Job.isRunning();
    const sam2::JobProgress prog = m_sam2Job.progress();

    ImGui::Begin("SAM2 Offline Pipeline");

    ImGui::Text("FPS: %.1f (%.2f ms)", m_fps, m_frameTimeMs);
    ImGui::Text("Resolution: %u x %u | Objects: %u | Textures: %u",
                m_renderer.getWidth(), m_renderer.getHeight(),
                (uint32_t)m_scene.getObjects().size(), m_textureManager.getTextureCount());
    ImGui::Text("Camera dist: %.2f | pitch: %.2f | yaw: %.2f",
                m_camera.distance, m_camera.pitch, m_camera.yaw);
    ImGui::Separator();

    // ─── Server settings (locked while a job runs) ───
    if (ImGui::CollapsingHeader("SAM2 Server", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::BeginDisabled(running);

        char hostBuf[128];
        std::snprintf(hostBuf, sizeof(hostBuf), "%s", m_sam2Host.c_str());
        if (ImGui::InputText("Host", hostBuf, sizeof(hostBuf)))
            m_sam2Host = hostBuf;

        ImGui::InputInt("Port", &m_sam2Port);
        m_sam2Port = std::clamp(m_sam2Port, 1, 65535);

        ImGui::SliderInt("Points per side", &m_sam2PointsPerSide, 0, 64);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = server default (32).\nLower = faster, coarser masks, less VRAM.");

        ImGui::EndDisabled();

        ImGui::BeginDisabled(m_sam2HealthPending);
        if (ImGui::Button(m_sam2HealthPending ? "Checking..." : "Check server"))
            startSam2HealthCheck();
        ImGui::EndDisabled();
    }

    // ─── Capture / segment ───
    if (ImGui::CollapsingHeader("Capture", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::BeginDisabled(running);
        char dirBuf[256];
        std::snprintf(dirBuf, sizeof(dirBuf), "%s", m_sam2OutputDir.c_str());
        if (ImGui::InputText("Output dir", dirBuf, sizeof(dirBuf)))
            m_sam2OutputDir = dirBuf;
        ImGui::EndDisabled();

        ImGui::BeginDisabled(running || m_sam2CaptureNow);
        if (ImGui::Button("Capture layers"))
            m_sam2CaptureNow = true;

        ImGui::SameLine();
        ImGui::BeginDisabled(!m_sam2LastCapture);
        if (ImGui::Button("Segment last capture"))
            startSam2Segmentation();
        ImGui::EndDisabled();
        ImGui::EndDisabled();

        ImGui::Checkbox("Auto-segment after capture", &m_sam2AutoSegment);
        ImGui::Checkbox("Save mask images", &m_sam2SaveMaskImages);

        if (running)
        {
            const float frac = prog.total ? float(prog.done) / float(prog.total) : 0.0f;
            char overlay[128];
            std::snprintf(overlay, sizeof(overlay), "%u / %u layers  (%.1f s)",
                          prog.done, prog.total, prog.elapsedSec);
            ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f), overlay);

            if (prog.currentLayer >= 0)
                ImGui::Text("SAM2 is processing layer %d ...", prog.currentLayer);

            if (ImGui::Button("Cancel"))
            {
                m_sam2Job.cancel();
                m_sam2Status = "Cancelling after the current layer...";
            }
        }
    }

    // ─── Union-find parameters used by the capture ───
    if (ImGui::CollapsingHeader("Union-Find (capture)"))
    {
        ImGui::TextDisabled("Applied on the next capture.");
        const char *mergeModes[] = {"Balanced", "Color priority", "Space priority"};
        int mode = (int)m_renderer.mergeMode;
        if (ImGui::Combo("Merge Mode", &mode, mergeModes, IM_ARRAYSIZE(mergeModes)))
            m_renderer.mergeMode = (uint32_t)mode;
        ImGui::SliderFloat("Plane Threshold", &m_renderer.planeThreshold, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("Proximity Scale", &m_renderer.proximityScale, 0.5f, 10.0f, "%.2f");
        ImGui::SliderFloat("Color Threshold", &m_renderer.colorThreshold, 0.05f, 1.5f, "%.2f");
        int iterations = (int)m_renderer.mergeIterations;
        if (ImGui::SliderInt("Merge Iterations", &iterations, 1, 8))
            m_renderer.mergeIterations = (uint32_t)iterations;
    }

    // ─── SAM2 vs union-find evaluation ───
    if (ImGui::CollapsingHeader("Evaluation: SAM2 vs Union-Find", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputInt("Min region pixels", &m_evalMinRegionPixels);
        m_evalMinRegionPixels = std::max(m_evalMinRegionPixels, 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Regions and masks smaller than this are left out of the\n"
                              "averages and match counts (their pixels still count in IoU).");

        ImGui::BeginDisabled(running || !m_sam2LastCapture || m_sam2Results.empty());
        if (ImGui::Button("Evaluate"))
            runSam2Evaluation();
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(runs automatically after segmentation)");

        if (m_hasEvaluation)
            drawEvaluationTable();
    }

    if (!m_sam2Status.empty())
    {
        ImGui::Separator();
        ImGui::TextWrapped("%s", m_sam2Status.c_str());
    }

    // ─── Results table ───
    if (!m_sam2Results.empty())
    {
        ImGui::Separator();
        const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
        if (ImGui::BeginTable("sam2_results", 6, flags))
        {
            ImGui::TableSetupColumn("Layer");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupColumn("Masks");
            ImGui::TableSetupColumn("SAM2 ms");
            ImGui::TableSetupColumn("Round trip ms");
            ImGui::TableSetupColumn("Upload KB");
            ImGui::TableHeadersRow();

            float sumInference = 0.0f, sumRoundTrip = 0.0f;
            uint32_t sumMasks = 0;

            for (const auto &r : m_sam2Results)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%d", r.layer);

                ImGui::TableSetColumnIndex(1);
                if (r.skipped)
                    ImGui::TextDisabled("empty");
                else if (r.ok)
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "ok");
                else if (!r.error.empty())
                {
                    const bool notProcessed = r.error.rfind("not processed", 0) == 0;
                    if (notProcessed)
                        ImGui::TextDisabled("not run");
                    else
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "error");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", r.error.c_str());
                }
                else if (running && prog.currentLayer == r.layer)
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "running...");
                else
                    ImGui::TextDisabled("pending");

                if (r.ok && !r.skipped)
                {
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%u", r.numMasks);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.1f", r.inferenceMs);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%.1f", r.roundTripMs);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.1f", r.requestBytes / 1024.0f);
                    sumMasks += r.numMasks;
                    sumInference += r.inferenceMs;
                    sumRoundTrip += r.roundTripMs;
                }
            }
            ImGui::EndTable();

            ImGui::Text("Total: %u masks | SAM2 %.2f s | round trip %.2f s",
                        sumMasks, sumInference / 1000.0f, sumRoundTrip / 1000.0f);
        }
    }

    ImGui::End();
}

void Application::drawEvaluationTable()
{
    if (m_evaluation.layers.empty())
    {
        ImGui::TextDisabled("No layer has a SAM2 result to compare.");
        return;
    }

    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("sam2_eval", 9, flags))
        return;

    ImGui::TableSetupColumn("Layer");
    ImGui::TableSetupColumn("UF regions");
    ImGui::TableSetupColumn("SAM masks");
    ImGui::TableSetupColumn("SAM unlabeled");
    ImGui::TableSetupColumn("UF best IoU");
    ImGui::TableSetupColumn("SAM best IoU");
    ImGui::TableSetupColumn("UF cover");
    ImGui::TableSetupColumn("Matches >0.5");
    ImGui::TableSetupColumn("F1@0.5");
    ImGui::TableHeadersRow();

    auto row = [](const char *label, const sam2::AgreementScore &s)
    {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1); ImGui::Text("%u", s.ufRegions);
        ImGui::TableSetColumnIndex(2); ImGui::Text("%u", s.samMasks);
        ImGui::TableSetColumnIndex(3); ImGui::Text("%.1f%%", s.samUnlabeled * 100.0f);
        ImGui::TableSetColumnIndex(4); ImGui::Text("%.3f", s.ufBestIoU);
        ImGui::TableSetColumnIndex(5); ImGui::Text("%.3f", s.samBestIoU);
        ImGui::TableSetColumnIndex(6); ImGui::Text("%.3f", s.ufCovering);
        ImGui::TableSetColumnIndex(7); ImGui::Text("%u (%.3f)", s.matches50, s.matchedIoU);
        ImGui::TableSetColumnIndex(8); ImGui::Text("%.3f", s.f1At50);
    };

    for (const auto &l : m_evaluation.layers)
    {
        char label[16];
        std::snprintf(label, sizeof(label), "%d", l.layer);
        row(label, l.score);
    }
    row("All", m_evaluation.total);
    ImGui::EndTable();

    ImGui::TextDisabled("Best IoU: per region, the best-overlapping region on the other side.\n"
                        "Matches >0.5: one-to-one region/mask pairs with IoU > 0.5 (mean IoU).");
}
#endif // RENDER_METHOD_SAM2
