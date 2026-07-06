#include "imgui_manager.h"
#include "vulkan_context.h"
#include "swapchain_manager.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <array>

bool ImGuiManager::init(VulkanContext& context, SwapchainManager& swapchain, GLFWwindow* window) {
    // Create descriptor pool for ImGui
    std::array<VkDescriptorPoolSize, 1> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
    }};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        std::cerr << "[ImGui] Failed to create descriptor pool" << std::endl;
        return false;
    }

    // Create render pass (LOAD existing content, end at PRESENT_SRC)
    if (!createRenderPass(context.device, swapchain.getImageFormat())) return false;
    if (!createFramebuffers(context.device, swapchain)) return false;

    // Init ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // Init GLFW backend
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Init Vulkan backend
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = context.instance;
    initInfo.PhysicalDevice = context.physical_device;
    initInfo.Device = context.device;
    initInfo.QueueFamily = context.graphics_queue_family;
    initInfo.Queue = context.graphics_queue;
    initInfo.DescriptorPool = m_descriptorPool;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = swapchain.getImageCount();

    initInfo.PipelineInfoMain.RenderPass = m_renderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);

    std::cout << "[ImGui] Initialized." << std::endl;
    return true;
}

void ImGuiManager::destroy(VkDevice device) {
    vkDeviceWaitIdle(device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    destroyFramebuffers(device);
    vkDestroyRenderPass(device, m_renderPass, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
}

void ImGuiManager::recreate(VulkanContext& context, SwapchainManager& swapchain) {
    destroyFramebuffers(context.device);
    createFramebuffers(context.device, swapchain);
}

void ImGuiManager::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiManager::render(VkCommandBuffer cmd, VkExtent2D extent, VkFramebuffer framebuffer) {
    ImGui::Render();

    VkRenderPassBeginInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpInfo.renderPass = m_renderPass;
    rpInfo.framebuffer = framebuffer;
    rpInfo.renderArea = {{0, 0}, extent};

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
}

bool ImGuiManager::createRenderPass(VkDevice device, VkFormat swapchainFormat) {
    VkAttachmentDescription att{};
    att.format = swapchainFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;    // Preserve composite output!
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // After composite
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;    // Ready for present

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1; rp.pAttachments = &att;
    rp.subpassCount = 1; rp.pSubpasses = &subpass;
    rp.dependencyCount = 1; rp.pDependencies = &dep;

    return vkCreateRenderPass(device, &rp, nullptr, &m_renderPass) == VK_SUCCESS;
}

bool ImGuiManager::createFramebuffers(VkDevice device, SwapchainManager& swapchain) {
    m_framebuffers.resize(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); i++) {
        VkImageView view = swapchain.getImageView(i);
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = m_renderPass;
        fb.attachmentCount = 1; fb.pAttachments = &view;
        fb.width = swapchain.getExtent().width;
        fb.height = swapchain.getExtent().height;
        fb.layers = 1;
        if (vkCreateFramebuffer(device, &fb, nullptr, &m_framebuffers[i]) != VK_SUCCESS) return false;
    }
    return true;
}

void ImGuiManager::destroyFramebuffers(VkDevice device) {
    for (auto fb : m_framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    m_framebuffers.clear();
}