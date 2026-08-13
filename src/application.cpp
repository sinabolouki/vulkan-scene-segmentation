#include "application.h"

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <array>
#include <cstring>
#include <imgui.h>

void Application::framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
    auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
    app->m_framebufferResized = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

bool Application::init(uint32_t width, uint32_t height, const std::string &title)
{
    m_width = width;
    m_height = height;

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
    // std::string scenePath = "../assets/circles16.obj";
    // std::string scenePath = "../assets/simple_scene.obj";
    // std::string scenePath = "assets/sponza/sponza.obj";
    std::string scenePath = "assets/simple_room/simple_room.obj";
    if (!m_scene.loadFromOBJ(scenePath))
    {
        std::cerr << "Failed to load scene: " << scenePath << std::endl;
        return false;
    }
    if (!uploadSceneToGPU())
        return false;

    glfwSetKeyCallback(m_window, keyCallback);

    // Textures (must happen BEFORE renderer init)
    if (!loadTextures())
        return false;

    // Give texture manager to renderer
    m_renderer.setTextureManager(&m_textureManager);

    // Renderer
    if (!m_renderer.init(m_context, m_swapchain))
        return false;

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
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Camera
    float aspect = (float)m_swapchain.getExtent().width / (float)m_swapchain.getExtent().height;
    glm::mat4 view = m_camera.getViewMatrix();
    m_renderer.nearPlane = 0.01f;
    m_renderer.farPlane = m_camera.distance * 10.0f;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, m_renderer.nearPlane, m_renderer.farPlane);
    glm::mat4 viewProj = proj * view;

    m_renderer.recordCommands(cmd, m_context, m_swapchain,
                              m_vertexBuffer, m_indexBuffer,
                              m_scene.getObjects(),
                              viewProj, view, proj, imageIndex);

    m_imguiManager.render(cmd, m_swapchain.getExtent(),
                          m_imguiManager.getFramebuffer(imageIndex));

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

    // Begin ImGui frame (before fence wait is fine — ImGui just builds draw lists)
    m_imguiManager.beginFrame();
    updateCamera();

    // ─── ImGui UI ───
     #ifdef RENDER_METHOD_UNION_FIND
    ImGui::Begin("Union-Find Renderer");

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
    ImGui::Separator();

    if (m_renderer.enableTiming) {
        auto& t = m_renderer.timingResults;
        ImGui::Text("GPU Timing (ms):");
        ImGui::Separator();
        ImGui::Text("  Clear:        %6.3f", t.clear);
        ImGui::Text("  Collect:      %6.3f", t.collect);
        ImGui::Text("  Sort:         %6.3f", t.sort);
        ImGui::Text("  Init Parents: %6.3f", t.initParents);
        ImGui::Text("  Merge:        %6.3f", t.merge);
        ImGui::Text("  Compress:     %6.3f", t.compress);
        ImGui::Separator();

        for (uint32_t i = 0; i < m_renderer.NUM_LAYERS; i++) {
            if (ImGui::TreeNode((void*)(intptr_t)i, "Peel %d: %6.3f ms", i, t.peels[i].total)) {
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

    const char *layerNames[] = {"Composite", "Layer 0", "Layer 1", "Layer 2", "Layer 3",
                                "Layer 4", "Layer 5", "Layer 6", "Layer 7"};
    int layerIdx = m_renderer.debugDisplayLayer + 1;
    ImGui::Combo("Display", &layerIdx, layerNames, 9);
    m_renderer.debugDisplayLayer = layerIdx - 1;

    ImGui::Separator();
    ImGui::Checkbox("Bypass Union-Find (show front only)", &m_renderer.bypassUnionFind);


    // ─── Debug Info ───
    if (ImGui::CollapsingHeader("Debug Info")) {
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
    //ImGui::SliderFloat("Depth Threshold", &m_renderer.depthThreshold, 0.001f, 0.1f, "%.4f");
    //ImGui::SliderFloat("Color Threshold", &m_renderer.colorThreshold, 0.01f, 0.5f, "%.3f");
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
#endif
    // ─── End ImGui UI ───

    VkDevice device = m_context.device;

    VkFence inFlightFence = m_swapchain.getInFlightFence(m_currentFrame);
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device, m_swapchain.getSwapchain(), UINT64_MAX,
        m_swapchain.getImageAvailableSemaphore(m_currentFrame),
        VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        m_swapchain.recreate(m_context, (uint32_t)w, (uint32_t)h);
        m_renderer.recreate(m_context, m_swapchain);
        m_imguiManager.recreate(m_context, m_swapchain);
        return;
    }

    vkResetFences(device, 1, &inFlightFence);
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = {m_swapchain.getImageAvailableSemaphore(m_currentFrame)};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signalSemaphores[] = {m_swapchain.getRenderFinishedSemaphore(m_currentFrame)};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
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
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        m_swapchain.recreate(m_context, (uint32_t)w, (uint32_t)h);
        m_renderer.recreate(m_context, m_swapchain);
        m_imguiManager.recreate(m_context, m_swapchain);
    }

    m_currentFrame = (m_currentFrame + 1) % SwapchainManager::MAX_FRAMES_IN_FLIGHT;
}

// Add the callback implementation:
void Application::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    #ifndef RENDER_METHOD_SIMPLE
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
    VkDevice device = m_context.device;
    vkDeviceWaitIdle(device);

    m_imguiManager.destroy(device);
    m_textureManager.destroy(m_context.device);
    m_renderer.destroy(device);

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

void Application::initCamera() {
    glm::vec3 center = m_scene.getCenter();
    float radius = m_scene.getRadius();

    // m_camera.target = center;
    m_camera.target = glm::vec3(-7.493f,5.453f,-3.652f);
    // m_camera.distance = radius * 0.1f;
    m_camera.distance = 2.74f;
    m_camera.yaw = 0.0f;
    m_camera.pitch = 0.3f;
    m_camera.updateFromOrbit();
}

void Application::updateCamera() {
    // Don't process camera input if ImGui wants the mouse
    if (ImGui::GetIO().WantCaptureMouse) {
        m_firstMouse = true;  // reset so we don't get a jump when returning to viewport
        return;
    }

    double mouseX, mouseY;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    if (m_firstMouse) {
        m_lastMouseX = mouseX;
        m_lastMouseY = mouseY;
        m_firstMouse = false;
    }

    double dx = mouseX - m_lastMouseX;
    double dy = mouseY - m_lastMouseY;
    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;

    // Left mouse: orbit
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        m_camera.yaw   -= float(dx) * m_camera.orbitSpeed;
        m_camera.pitch += float(dy) * m_camera.orbitSpeed;
        m_camera.pitch = glm::clamp(m_camera.pitch, -1.5f, 1.5f);
        m_camera.updateFromOrbit();
    }

    // Middle mouse: pan
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
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
    if (scroll != 0.0f) {
        m_camera.distance -= scroll * m_camera.distance * m_camera.zoomSpeed;
        m_camera.distance = glm::max(m_camera.distance, 0.01f);
        m_camera.updateFromOrbit();
    }
}

bool Application::loadTextures() {
    if (!m_textureManager.init(m_context)) {
        std::cerr << "[App] Failed to init texture manager" << std::endl;
        return false;
    }

    auto texturePaths = m_scene.getTextureFilenames();
    std::cout << "[App] Found " << texturePaths.size() << " textures to load" << std::endl;

    if (!texturePaths.empty()) {
        if (!m_textureManager.loadTextures(m_context, texturePaths)) {
            std::cerr << "[App] Failed to load textures" << std::endl;
            return false;
        }
        m_scene.resolveTextureIndices(m_textureManager.getLoadedPaths());
    }

    return true;
}