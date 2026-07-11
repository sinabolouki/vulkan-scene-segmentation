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
    std::string scenePath = "../assets/simple_scene.obj";
    if (!m_scene.loadFromOBJ(scenePath))
    {
        std::cerr << "Failed to load scene: " << scenePath << std::endl;
        return false;
    }
    if (!uploadSceneToGPU())
        return false;

    glfwSetKeyCallback(m_window, keyCallback);

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
    glm::vec3 center = m_scene.getCenter();
    float radius = m_scene.getRadius();
    glm::vec3 eye = center + glm::vec3(0.0f, radius * 0.5f, radius * 2.0f);
    eye = center + glm::vec3(1.5f, 0.5f, 0.0f) * radius; // Slightly offset for better view
    glm::mat4 view = glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f));
    m_renderer.nearPlane = 0.01f;
    m_renderer.farPlane = radius * 10.0f;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, m_renderer.nearPlane, m_renderer.farPlane);
    glm::mat4 viewProj = proj * view;

    // Both renderers share the same interface:
    m_renderer.recordCommands(cmd, m_context, m_swapchain,
                              m_vertexBuffer, m_indexBuffer,
                              m_scene.getObjects(),
                              viewProj, imageIndex);

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

    // ─── ImGui UI ───
    ImGui::Begin("PPLL Renderer");
    ImGui::Text("FPS: %.1f (%.2f ms)", m_fps, m_frameTimeMs);
    ImGui::Separator();
    ImGui::SliderFloat("Depth Threshold", &m_renderer.depthThreshold, 0.001f, 0.1f, "%.4f");
    ImGui::SliderFloat("Color Threshold", &m_renderer.colorThreshold, 0.01f, 0.5f, "%.3f");
#ifdef RENDER_METHOD_PPLL
    int grow = (int)m_renderer.growIterations;
    ImGui::SliderInt("Grow Iterations", &grow, 1, 100);
    m_renderer.growIterations = (uint32_t)grow;
#endif
    ImGui::Separator();
    const char *layerNames[] = {"Composite", "Layer 0", "Layer 1", "Layer 2", "Layer 3"};
    int layerIdx = m_renderer.debugDisplayLayer + 1; // -1 → 0, 0→1, etc.
    ImGui::Combo("Display", &layerIdx, layerNames, 5);
    m_renderer.debugDisplayLayer = layerIdx - 1;
    ImGui::End();
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
        case GLFW_KEY_5:
            app->m_renderer.debugDisplayLayer = 100;
            break; // Root colors
        case GLFW_KEY_6:
            app->m_renderer.debugDisplayLayer = 101;
            break; // Root count
        case GLFW_KEY_7:
            app->m_renderer.debugDisplayLayer = 102;
            break; // Depth
        case GLFW_KEY_8:
            app->m_renderer.debugDisplayLayer = 103;
            break; // Frag count
        case GLFW_KEY_9:
            app->m_renderer.debugDisplayLayer = 104;
            break; // Bypass test
        }
    }
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