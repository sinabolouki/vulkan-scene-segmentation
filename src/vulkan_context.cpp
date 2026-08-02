#include "vulkan_context.h"
#include <iostream>

bool VulkanContext::init(GLFWwindow* window) {
    // ─── Instance ───
    vkb::InstanceBuilder instance_builder;
    auto instance_result = instance_builder
        .set_app_name("DepthPeelingRenderer")
        .request_validation_layers(true)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    if (!instance_result) {
        std::cerr << "Failed to create Vulkan instance: "
                  << instance_result.error().message() << std::endl;
        return false;
    }

    vkb::Instance vkb_instance = instance_result.value();
    instance = vkb_instance.instance;
    debug_messenger = vkb_instance.debug_messenger;

    // ─── Surface ───
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        std::cerr << "Failed to create window surface" << std::endl;
        return false;
    }

    VkPhysicalDeviceFeatures features{};
    features.fragmentStoresAndAtomics = VK_TRUE;
    features.samplerAnisotropy = VK_TRUE;  // ← FIX: needed for texture sampler

    // ─── Physical Device ───
    vkb::PhysicalDeviceSelector phys_selector(vkb_instance);
    auto phys_result = phys_selector
        .set_surface(surface)
        .set_minimum_version(1, 3)
        .set_required_features(features)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        // ← FIX: removed require_dedicated_transfer_queue()
        .select();

    if (!phys_result) {
        std::cerr << "Failed to select physical device: "
                  << phys_result.error().message() << std::endl;
        return false;
    }

    vkb::PhysicalDevice vkb_physical_device = phys_result.value();
    physical_device = vkb_physical_device.physical_device;

    // ─── Logical Device ───
    vkb::DeviceBuilder device_builder(vkb_physical_device);
    auto device_result = device_builder.build();

    if (!device_result) {
        std::cerr << "Failed to create logical device: "
                  << device_result.error().message() << std::endl;
        return false;
    }

    vkb::Device vkb_device = device_result.value();
    device = vkb_device.device;

    // ─── Queues ───
    auto gq = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!gq) {
        std::cerr << "Failed to get graphics queue" << std::endl;
        return false;
    }
    graphics_queue = gq.value();
    graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    auto pq = vkb_device.get_queue(vkb::QueueType::present);
    if (!pq) {
        std::cerr << "Failed to get present queue" << std::endl;
        return false;
    }
    present_queue = pq.value();

    // ─── Command Pool ───  ← FIX: create a general-purpose command pool
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphics_queue_family;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool in VulkanContext" << std::endl;
        return false;
    }

    std::cout << "Vulkan context initialized successfully." << std::endl;
    std::cout << "GPU: " << vkb_physical_device.name << std::endl;

    return true;
}

void VulkanContext::destroy() {
    if (command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, command_pool, nullptr);
    }
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }
    vkb::destroy_debug_utils_messenger(instance, debug_messenger);
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}