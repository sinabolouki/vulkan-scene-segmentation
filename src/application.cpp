#include "application.h"

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <array>
#include <cstring>

// Static Callback
void Application::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_framebufferResized = true;
}

// Init
bool Application::init(uint32_t width, uint32_t height, const std::string& title) {
    m_width = width;
    m_height = height;

    // GLFW
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) return false;
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);

    // Vulkan context (instance, device, queues)
    if (!m_context.init(m_window)) return false;

    // Swapchain (images, render pass, framebuffers, sync)
    if (!m_swapchain.init(m_context, width, height)) return false;

    // Commands
    if (!createCommandPool()) return false;
    if (!createCommandBuffers()) return false;

    // Pipeline
    if (!createPipelineLayout()) return false;
    if (!createPipelines()) return false;

    // Geometry
    if (!createSceneGeometry()) return false;

    std::cout << "Application fully initialized." << std::endl;
    return true;
}

// Pipeline Layout
bool Application::createPipelineLayout() {
    // Push constants: MVP matrix + objectID
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(ScenePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;           // No descriptor sets yet
    layoutInfo.pSetLayouts = nullptr;        // Will add for peel pass (prev depth sampler)
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(m_context.device, &layoutInfo, nullptr, &m_scenePipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }

    return true;
}

// Pipelines
bool Application::createPipelines() {
    // Load shaders
    VkShaderModule vertModule = ShaderUtils::createShaderModule(m_context.device, "shaders/geometry.vert.spv");
    VkShaderModule fragModule = ShaderUtils::createShaderModule(m_context.device, "shaders/depth_init.frag.spv");

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }

    // Get vertex input descriptions from Vertex struct
    VkVertexInputBindingDescription binding = Vertex::getBindingDescription();
    std::vector<VkVertexInputAttributeDescription> attributes = Vertex::getAttributeDescriptions();

    // Build the init pipeline (standard forward rendering with depth)
    PipelineBuilder builder;
    m_initPipeline = builder
        .setVertexShader(vertModule)
        .setFragmentShader(fragModule)
        .setVertexInput({binding}, attributes)
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .enableDepthTest(true, VK_COMPARE_OP_LESS)
        .disableBlending()
        .setPipelineLayout(m_scenePipelineLayout)
        .setRenderPass(m_swapchain.getRenderPass(), 0)
        .build(m_context.device);

    // Destroy shader modules (safe after pipeline is created)
    vkDestroyShaderModule(m_context.device, vertModule, nullptr);
    vkDestroyShaderModule(m_context.device, fragModule, nullptr);

    if (m_initPipeline == VK_NULL_HANDLE) {
        std::cerr << "Failed to build init pipeline" << std::endl;
        return false;
    }

    std::cout << "Pipelines created successfully." << std::endl;
    return true;
}

// Scene Geometry — Two overlapping quads
bool Application::createSceneGeometry() {
    // Two overlapping quads at different depths
    // Quad 1: red, at z = -1.0
    // Quad 2: blue, at z = -1.5 (partially behind quad 1)
    std::vector<Vertex> vertices = {
        // Quad 1 (red)
        { {-0.5f, -0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f} },
        { { 0.5f, -0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f} },
        { { 0.5f,  0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f} },
        { {-0.5f,  0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f} },

        // Quad 2 (blue, offset to overlap)
        { { 0.0f, -0.3f, -1.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f} },
        { { 1.0f, -0.3f, -1.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f} },
        { { 1.0f,  0.7f, -1.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f} },
        { { 0.0f,  0.7f, -1.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f} },
    };

    std::vector<uint32_t> indices = {
        // Quad 1
        0, 1, 2,  2, 3, 0,
        // Quad 2
        4, 5, 6,  6, 7, 4,
    };

    m_indexCount = static_cast<uint32_t>(indices.size());

    // ─── Vertex Buffer ───
    VkDeviceSize vertexSize = sizeof(Vertex) * vertices.size();
    if (!createBuffer(vertexSize,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_vertexBuffer, m_vertexBufferMemory)) {
        return false;
    }

    void* data;
    vkMapMemory(m_context.device, m_vertexBufferMemory, 0, vertexSize, 0, &data);
    memcpy(data, vertices.data(), vertexSize);
    vkUnmapMemory(m_context.device, m_vertexBufferMemory);

    // ─── Index Buffer ───
    VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
    if (!createBuffer(indexSize,
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_indexBuffer, m_indexBufferMemory)) {
        return false;
    }

    vkMapMemory(m_context.device, m_indexBufferMemory, 0, indexSize, 0, &data);
    memcpy(data, indices.data(), indexSize);
    vkUnmapMemory(m_context.device, m_indexBufferMemory);

    std::cout << "Scene geometry created: " << vertices.size() << " vertices, "
              << indices.size() << " indices." << std::endl;
    return true;
}

// Record Command Buffer — Render geometry with init pipeline
void Application::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Begin render pass
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { {0.02f, 0.02f, 0.08f, 1.0f} };
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_swapchain.getRenderPass();
    rpInfo.framebuffer = m_swapchain.getFramebuffer(imageIndex);
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = m_swapchain.getExtent();
    rpInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    // ─── Bind Pipeline ───
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_initPipeline);

    // ─── Set Dynamic Viewport & Scissor ───
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchain.getExtent().width);
    viewport.height = static_cast<float>(m_swapchain.getExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchain.getExtent();
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // ─── Bind Vertex & Index Buffers ───
    VkBuffer vertexBuffers[] = {m_vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // ─── Draw Quad 1 (objectID = 0) ───
    {
        ScenePushConstants pc{};
        pc.model = glm::mat4(1.0f); // identity
        pc.viewProj = glm::perspective(glm::radians(60.0f),
                          (float)m_swapchain.getExtent().width / (float)m_swapchain.getExtent().height,
                          0.1f, 100.0f)
                      * glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        pc.viewProj[1][1] *= -1; // Invert Y for Vulkan's coordinate system
        pc.objectID = 0;

        vkCmdPushConstants(cmd, m_scenePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(ScenePushConstants), &pc);

        // Draw first quad: 6 indices starting at index 0
        vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }

    // ─── Draw Quad 2 (objectID = 1) ───
    {
        ScenePushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.viewProj = glm::perspective(glm::radians(60.0f),
                          (float)m_swapchain.getExtent().width / (float)m_swapchain.getExtent().height,
                          0.1f, 100.0f)
                      * glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        pc.viewProj[1][1] *= -1; // Invert Y for Vulkan's coordinate system
        pc.objectID = 1;

        vkCmdPushConstants(cmd, m_scenePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(ScenePushConstants), &pc);

        // Draw second quad: 6 indices starting at index 6
        vkCmdDrawIndexed(cmd, 6, 1, 6, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

// Draw Frame
void Application::drawFrame() {
    VkDevice device = m_context.device;

    VkFence inFlightFence = m_swapchain.getInFlightFence(m_currentFrame);
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device, m_swapchain.getSwapchain(), UINT64_MAX,
        m_swapchain.getImageAvailableSemaphore(m_currentFrame),
        VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        m_swapchain.recreate(m_context, (uint32_t)w, (uint32_t)h);
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

    if (vkQueueSubmit(m_context.graphics_queue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
        std::cerr << "Failed to submit draw command buffer" << std::endl;
    }

    result = m_swapchain.present(m_context.present_queue, imageIndex, signalSemaphores[0]);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
        m_framebufferResized = false;
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        m_swapchain.recreate(m_context, (uint32_t)w, (uint32_t)h);
    }

    m_currentFrame = (m_currentFrame + 1) % SwapchainManager::MAX_FRAMES_IN_FLIGHT;
}

// Main Loop
void Application::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(m_context.device);
}

// Buffer Creation Helper
bool Application::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags properties,
                               VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_context.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_context.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory" << std::endl;
        return false;
    }

    vkBindBufferMemory(m_context.device, buffer, memory, 0);
    return true;
}

uint32_t Application::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_context.physical_device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cerr << "Failed to find suitable memory type!" << std::endl;
    return 0;
}

// Command Pool & Buffers
bool Application::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_context.graphics_queue_family;

    if (vkCreateCommandPool(m_context.device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool Application::createCommandBuffers() {
    m_commandBuffers.resize(SwapchainManager::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

    if (vkAllocateCommandBuffers(m_context.device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
        return false;
    }
    return true;
}

// Cleanup
void Application::cleanup() {
    VkDevice device = m_context.device;

    // Geometry
    vkDestroyBuffer(device, m_indexBuffer, nullptr);
    vkFreeMemory(device, m_indexBufferMemory, nullptr);
    vkDestroyBuffer(device, m_vertexBuffer, nullptr);
    vkFreeMemory(device, m_vertexBufferMemory, nullptr);

    // Pipeline
    vkDestroyPipeline(device, m_initPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_scenePipelineLayout, nullptr);

    // Commands
    vkDestroyCommandPool(device, m_commandPool, nullptr);

    // Swapchain
    m_swapchain.destroy(device);

    // Context
    m_context.destroy();

    // Window
    glfwDestroyWindow(m_window);
    glfwTerminate();
}