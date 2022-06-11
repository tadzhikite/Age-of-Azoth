#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cmath>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <complex>
#include <set>
#include <numeric>
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << "\n\n";
    return VK_FALSE;
}
const auto framebufferResizeCallback = [](GLFWwindow* window, int width, int height) {};
struct Instance {
    std::vector<const char*> validationLayers;
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
};
Instance instanceVK = [](){
    auto inst = Instance{
        .validationLayers  = std::vector<const char*> { "VK_LAYER_KHRONOS_validation" },
            .debugCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = debugCallback
            }
    };
    const auto enable = inst.validationLayers.size()>=1;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    uint32_t count = 0;
    const char** e = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> exts(e, e + count);
    if (enable) {exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);}
    auto appInfo = VkApplicationInfo {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO
            ,   .pApplicationName = "Hello Triangle"
            ,   .applicationVersion = VK_MAKE_VERSION(1, 0, 0)
            ,   .pEngineName = "No Engine"
            ,   .engineVersion = VK_MAKE_VERSION(1, 0, 0)
            ,   .apiVersion = VK_API_VERSION_1_0 
    };
    auto instanceInfo = VkInstanceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = enable?(VkDebugUtilsMessengerCreateInfoEXT*) &inst.debugCreateInfo:nullptr,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = enable?static_cast<uint32_t>(inst.validationLayers.size()):0,
            .ppEnabledLayerNames = enable?inst.validationLayers.data():nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(exts.size()),
            .ppEnabledExtensionNames = exts.data(),
    }; 
    if(vkCreateInstance(&instanceInfo,nullptr,&inst.instance)!=VK_SUCCESS){throw std::runtime_error("failed instance");}
    if(enable&&[&inst](){ //vkCreateDebugUtilsMessengerEXT
            auto f=(PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(inst.instance,"vkCreateDebugUtilsMessengerEXT");
            return (f != nullptr)?f(inst.instance, &inst.debugCreateInfo, nullptr, &inst.debugMessenger):VK_ERROR_EXTENSION_NOT_PRESENT;
            }() 
            != VK_SUCCESS) { throw std::runtime_error("failed debug messenger!"); }
    return inst;
}();
struct Pipeline {
    struct PushConstant {
        std::array<float,16> transform;
        std::array<int, 16> index;
        static constexpr auto pushRange = VkPushConstantRange {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = 128,
        };
    };
    struct Vertex {
        glm::vec4 position;
        glm::vec4 color;
        glm::vec2 textureCoordinates;
    };
    struct Descriptor {
        std::pair<VkImageViewCreateInfo, VkDeviceMemory> texture;
        VkDescriptorImageInfo imageInfo;
        std::pair<VkDescriptorBufferInfo, VkDeviceMemory> uniform;
        VkDescriptorSetLayout dSetLayout;
        VkDescriptorSetAllocateInfo allocInfo;
        VkDescriptorSet dSet;
    };
    struct Attribute {
        std::pair<VkBuffer, VkDeviceMemory> vertices;
        std::pair<VkBuffer, VkDeviceMemory> indices;
        uint32_t size;
    };
    struct Parameter {
        //used for copy commands
    };
    std::vector<VkPipelineShaderStageCreateInfo>     shaderStages;
    VkVertexInputBindingDescription                  inputBinding;
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions;
    VkPipelineVertexInputStateCreateInfo             vertexInput;
    VkPipelineInputAssemblyStateCreateInfo           inputAssembly;
    VkViewport                                       view;
    VkRect2D                                         scissor;
    VkPipelineViewportStateCreateInfo                viewportState;
    VkPipelineRasterizationStateCreateInfo           rasterizer;
    VkPipelineMultisampleStateCreateInfo             multisample;
    VkPipelineColorBlendAttachmentState              colorBlendAttachment;
    VkPipelineDepthStencilStateCreateInfo            depthStencilState;
    VkPipelineColorBlendStateCreateInfo              colorBlend;
    std::vector<VkDynamicState>                      dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo                 dynamicState;
    VkGraphicsPipelineCreateInfo                     info;
    VkPipeline                                       pipe;
    Attribute                                        attribute;
    Descriptor                                       descriptor;
    Parameter                                        parameter;
    PushConstant                                     pushConstant;
    int                                              indexSize;
};
struct SwapchainSurface {
    GLFWwindow* window;
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    static constexpr auto SwapchainImages = [](const auto& dev, const auto& swap){
        std::vector<VkImage> swapImgs;
        unsigned int imageCount;
        vkGetSwapchainImagesKHR(dev, swap, &imageCount, nullptr);
        swapImgs.resize(imageCount);
        vkGetSwapchainImagesKHR(dev, swap, &imageCount, swapImgs.data());
        return swapImgs;
    };
    static constexpr auto SwapchainImageViews = [](const auto& dev, const auto& imgs, const auto& format) {
        std::vector<VkImageView> imgViews(imgs.size());
        std::transform(imgs.begin(), imgs.end(), imgViews.begin(), [&dev, &format](auto& sImg){
                auto info = VkImageViewCreateInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = sImg,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .components = VkComponentMapping{
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY },
                .subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1 }
                };
                VkImageView imgView;
                if (vkCreateImageView(dev, &info, nullptr, &imgView) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
                }
                return imgView;
        });
        return imgViews;
    };
    static constexpr auto DepthImage = [](const auto& dev, const auto& extent){
        const auto imgInfo = VkImageCreateInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_D24_UNORM_S8_UINT,
                .extent = VkExtent3D {
                    .width = static_cast<uint32_t>(extent.width),
                    .height = static_cast<uint32_t>(extent.height),
                    .depth = 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED 
        };
        VkImage image;
        if (vkCreateImage(dev, &imgInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }
        return image;
    };
    static constexpr auto DepthImageView = [](const auto& dev, const auto& img){
        const auto viewInfo = VkImageViewCreateInfo {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_D24_UNORM_S8_UINT,
                .subresourceRange = VkImageSubresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
        };
        VkImageView imgView;
        if (vkCreateImageView(dev, &viewInfo, nullptr, &imgView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }
        return imgView;
    };
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkSurfaceFormatKHR format;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;
    std::vector<VkFramebuffer> framebuffers;

    std::vector<VkImageView> viewAttachments;
    std::vector<VkFramebufferCreateInfo> framebufferInfo;
    void Construct(const VkPhysicalDevice& pDev, const VkDevice& dev, const Pipeline& pipe){
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDev, info.surface, &surfaceCapabilities);
        auto c = surfaceCapabilities;

        std::vector<VkQueueFamilyProperties> qFams;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pDev, &count, nullptr);
        qFams.resize(count);
        vkGetPhysicalDeviceQueueFamilyProperties(pDev, &count, qFams.data());
        auto qfi = std::vector<uint32_t>{ static_cast<uint32_t>(std::find_if(qFams.begin(), qFams.end(), 
                    [](const auto& f) { return (f.queueFlags & VK_QUEUE_GRAPHICS_BIT);}
                    )-qFams.begin()),
             static_cast<uint32_t>(std::find_if(qFams.begin(), qFams.end(),
                         [this, &pDev, i = 0](const auto& f) mutable {
                         VkBool32 res = false;
                         vkGetPhysicalDeviceSurfaceSupportKHR(pDev, i++, info.surface, &res);
                         return res && !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT); })-qFams.begin())
        };
        info = VkSwapchainCreateInfoKHR {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .surface = info.surface,
                .minImageCount=(c.maxImageCount>0&&c.minImageCount+1>c.maxImageCount)?c.maxImageCount:c.minImageCount+1,
                .imageFormat = format.format,
                .imageColorSpace = format.colorSpace,
                .imageExtent = [this](const auto& capabilities){
                    auto min = capabilities.minImageExtent;
                    auto max = capabilities.maxImageExtent;
                    auto cur = capabilities.currentExtent;
                    int width, height;
                    glfwGetFramebufferSize(window, &width, &height);
                    auto w = static_cast<uint32_t>(width);
                    auto h = static_cast<uint32_t>(height);
                    return (cur.width == UINT32_MAX)?
                        VkExtent2D{
                            .width = glm::clamp(w, min.width, max.width), 
                                .height = glm::clamp(h, min.height, max.height) 
                        }:cur;
                }(surfaceCapabilities),
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = static_cast<uint32_t>(qfi.size()),
                .pQueueFamilyIndices = qfi.data(),
                .preTransform = c.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = [&pDev](const auto& surf){
                    uint32_t count = 0;
                    vkGetPhysicalDeviceSurfacePresentModesKHR(pDev, surf, &count, nullptr);
                    std::vector<VkPresentModeKHR> presentModes(count);
                    vkGetPhysicalDeviceSurfacePresentModesKHR(pDev, surf, &count, presentModes.data());
                    auto res = std::find(presentModes.begin(), presentModes.end(), VK_PRESENT_MODE_MAILBOX_KHR);
                    return res==presentModes.end()?VK_PRESENT_MODE_FIFO_KHR:*res;
                }(info.surface),
                .clipped = VK_TRUE,
        };
        if (vkCreateSwapchainKHR(dev, &info, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
        depthImage = DepthImage(dev, info.imageExtent);
        depthImageMemory = [&pDev](const auto& dev, const auto& img){
            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(dev, img, &memRequirements);
            const auto allocInfo = VkMemoryAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = memRequirements.size,
                    .memoryTypeIndex = [&pDev](const auto& t, const auto& p){
                        VkPhysicalDeviceMemoryProperties mP;
                        vkGetPhysicalDeviceMemoryProperties(pDev, &mP);
                        for (uint32_t i = 0; i < mP.memoryTypeCount; i++) {
                            if ((t & (1 << i)) && (mP.memoryTypes[i].propertyFlags & p) == p) {
                                return i;
                            }
                        }
                        throw std::runtime_error("failed to find suitable memory type!");
                    }(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            };
            VkDeviceMemory imageMemory;
            if (vkAllocateMemory(dev, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate image memory!");
            }
            vkBindImageMemory(dev, img, imageMemory, 0);
            return imageMemory;
        }(dev, depthImage);
        depthImageView = DepthImageView(dev, depthImage);
        images = SwapchainImages(dev, swapchain);
        imageViews = SwapchainImageViews(dev, images, format.format);
        auto SwapchainFramebuffers = [this, &pipe](const auto& dev, const auto& imgViews, const auto& depthView){
            std::vector<VkFramebuffer> fBs(imgViews.size());
            viewAttachments.resize(2*imgViews.size());
            framebufferInfo.resize(imgViews.size());
            for(size_t i = 0; i < imgViews.size(); i++){
                viewAttachments[2*i] = imgViews[i];
                viewAttachments[2*i+1] = depthView;
                framebufferInfo[i] = VkFramebufferCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                        .renderPass = pipe.info.renderPass,
                        .attachmentCount = 2,
                        .pAttachments = &viewAttachments[2*i],
                        .width = info.imageExtent.width,
                        .height = info.imageExtent.height,
                        .layers = 1 
                };
                if (vkCreateFramebuffer(dev, &framebufferInfo[i], nullptr, &fBs[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create framebuffer!");
                }

            }
            return fBs;
        };
        framebuffers = SwapchainFramebuffers(dev, imageViews, depthImageView);
    };
    void resize(const VkPhysicalDevice& pDev, const VkDevice& dev, const Pipeline& pipe) {
        vkDeviceWaitIdle(dev);
        vkDestroyImage(dev, depthImage, nullptr);
        vkFreeMemory(dev, depthImageMemory, nullptr);
        vkDestroyImageView(dev, depthImageView, nullptr);
        vkDestroySwapchainKHR(dev, swapchain, nullptr);
        for (auto view : imageViews) { vkDestroyImageView(dev, view, nullptr); }
        for (auto buffer : framebuffers) { vkDestroyFramebuffer(dev, buffer, nullptr); }
        Construct(pDev, dev, pipe);
    };
};
auto swap = [](){
    auto swapchain = SwapchainSurface{ .window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr) };
    swapchain.info = VkSwapchainCreateInfoKHR{ .surface = [&swapchain](){ VkSurfaceKHR s;
        if (glfwCreateWindowSurface(instanceVK.instance, swapchain.window, nullptr, &s) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        return s;
    }() };
    return swapchain;
}();
const std::vector<const char*> devExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
auto physicalDevice = [](const auto& inst, const auto& surf){
    auto devs = [&inst]() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(inst, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> d(deviceCount);
        vkEnumeratePhysicalDevices(inst, &deviceCount, d.data());
        return d;
    }();
    return *std::find_if(devs.begin(),devs.end(),[&surf](const VkPhysicalDevice& pDev){
            uint32_t f = 0;
            uint32_t p = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(pDev, surf, &f, nullptr);
            vkGetPhysicalDeviceSurfacePresentModesKHR(pDev, surf, &p, nullptr);
            auto queueFamilies = [](const auto& device) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
            return queueFamilies;
            }(pDev);
            const auto av = [](auto& pDev) {
            std::vector<VkExtensionProperties> exts;
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(pDev, nullptr, &extensionCount, nullptr);
            exts.resize(extensionCount);
            vkEnumerateDeviceExtensionProperties(pDev, nullptr, &extensionCount, exts.data());
            return exts;
            }(pDev);
            return f!=0&&p!=0&&std::accumulate(devExts.begin(), devExts.end(), true, [&av](bool a, const char* rExt){
                    return a && std::find_if(av.begin(), av.end(), 
                            [&rExt](const VkExtensionProperties& e) {
                            return strcmp(rExt, e.extensionName) == 0;}
                            )!=av.end(); }) && std::find_if(queueFamilies.begin(), queueFamilies.end(), 
                            [](const VkQueueFamilyProperties& f) { 
                            return (f.queueFlags & VK_QUEUE_GRAPHICS_BIT);
                            })!=std::end(queueFamilies) && std::find_if(queueFamilies.begin(), queueFamilies.end(),
                                [&pDev, &surf, i = 0](const VkQueueFamilyProperties& f) {
                                VkBool32 s = false;
                                vkGetPhysicalDeviceSurfaceSupportKHR(pDev, i, surf, &s);
                                return s & !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT); 
                                })!=std::end(queueFamilies);
    });
}(instanceVK.instance, swap.info.surface);
const auto QueueFamilyGST = [](){
    std::vector<VkQueueFamilyProperties> queueFamilies;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    queueFamilies.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, queueFamilies.data());
    uint32_t graphicsFamily = std::find_if(queueFamilies.begin(), queueFamilies.end(), 
            [](const auto& f) { return (f.queueFlags & VK_QUEUE_GRAPHICS_BIT);}
            )-queueFamilies.begin();
    uint32_t surfaceFam = std::find_if(queueFamilies.begin(), queueFamilies.end(),
            [i = 0](const auto& f) mutable {
            VkBool32 res = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i++, swap.info.surface, &res);
            return res && !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT); })-queueFamilies.begin();
    uint32_t transferFamily = std::find_if(queueFamilies.begin(), queueFamilies.end(), 
            [](const auto& f) { return (f.queueFlags & VK_QUEUE_TRANSFER_BIT)&&!(f.queueFlags & VK_QUEUE_GRAPHICS_BIT);}
            )-queueFamilies.begin();
    return std::array<uint32_t, 3> {graphicsFamily, surfaceFam, transferFamily };
}();
auto device = [](){ 
    std::vector<VkDeviceQueueCreateInfo> queueInfo;
    float qP = 1.0f;
    std::transform(QueueFamilyGST.begin(),QueueFamilyGST.end(),std::back_inserter(queueInfo),
            [&qP](const uint32_t it){ return VkDeviceQueueCreateInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = it,
            .queueCount = 1,
            .pQueuePriorities = &qP}; 
            });
    VkPhysicalDeviceFeatures deviceFeatures{ .samplerAnisotropy = VK_TRUE };
    auto info = VkDeviceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueInfo.size()),
            .pQueueCreateInfos = queueInfo.data(),
            .enabledExtensionCount = static_cast<uint32_t>(devExts.size()),
            .ppEnabledExtensionNames = devExts.data(),
            .pEnabledFeatures = &deviceFeatures };
    VkDevice d;
    if (vkCreateDevice(physicalDevice, &info, nullptr, &d) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    return d;
}();
float programClock = 0.f;
struct SyncPrims{
    VkSemaphore a;
    VkSemaphore b;
    VkFence f;
    std::array<VkResult, 3> results;
    SyncPrims() = default;
    SyncPrims(const VkDevice& d){
        auto sI = VkSemaphoreCreateInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        auto fI = VkFenceCreateInfo{ 
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 
                .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        results = std::array<VkResult, 3>{
            vkCreateSemaphore(d, &sI, nullptr, &a),
                vkCreateSemaphore(d, &sI, nullptr, &b),
                vkCreateFence(d, &fI, nullptr, &f)}; };
    void Destroy(const VkDevice& d){
        vkDestroySemaphore(d, a, nullptr);
        vkDestroySemaphore(d, b, nullptr);
        vkDestroyFence(d, f, nullptr);
    }
};
auto syncs = [](const auto& dev){
    std::array<SyncPrims, 2> s;
    std::generate(s.begin(), s.end(), [&dev](){ return SyncPrims(dev); });
    return s;
}(device);
std::vector<Pipeline> pipes;
auto PrimaryPipe = [](const auto& dev){
    auto pipeline = Pipeline{};
    pipeline.descriptor = [](){
        Pipeline::Descriptor descriptor{
            .texture = []() {
                stbi_uc* pixels;
                int width, height, channels;
                pixels = stbi_load("textures/8k_earth_daymap.jpg", &width, &height, &channels, STBI_rgb_alpha);
                if (!pixels) {
                    throw std::runtime_error("failed to load texture image!");
                }
                auto imgViewInfo = VkImageViewCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .image = [](const auto& dev, const auto& extent){
                            VkImageCreateInfo imgInfo{
                                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_SRGB,
                                .extent = extent,
                                .mipLevels = 1,
                                .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                .sharingMode = VK_SHARING_MODE_CONCURRENT,
                                .queueFamilyIndexCount = 3,
                                .pQueueFamilyIndices = &QueueFamilyGST[0],
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                            };
                            VkImage image;
                            if (vkCreateImage(dev, &imgInfo, nullptr, &image) != VK_SUCCESS) {
                                throw std::runtime_error("failed to create image!");
                            }
                            return image;
                        }(device, VkExtent3D{.width=static_cast<uint32_t>(width),.height=static_cast<uint32_t>(height),.depth = 1}),
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = VK_FORMAT_R8G8B8A8_SRGB,
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                        }
                };
                VkDeviceSize imageSize = width * height * 4;
                auto stagingBuffer = [](const auto& dev, const auto& size){
                    auto info = VkBufferCreateInfo {
                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = size,
                            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                            .queueFamilyIndexCount = 1,
                            .pQueueFamilyIndices = &QueueFamilyGST[2]
                    };
                    VkBuffer b;
                    if (vkCreateBuffer(dev, &info, nullptr, &b) != VK_SUCCESS) {
                        throw std::runtime_error("failed to create buffer!");
                    }
                    return b;
                }(device, imageSize);
                auto stagingBufferMemory = [](const auto& dev, const auto& buffer){
                    VkMemoryRequirements memReqs;
                    vkGetBufferMemoryRequirements(dev, buffer, &memReqs);
                    auto memAllocInfo = VkMemoryAllocateInfo {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                            .allocationSize = memReqs.size,
                            .memoryTypeIndex = [](const auto& memType){
                                const auto p = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                                VkPhysicalDeviceMemoryProperties mP;
                                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mP);
                                for (uint32_t i = 0; i < mP.memoryTypeCount; i++) {
                                    if ((memType & (1 << i)) && (mP.memoryTypes[i].propertyFlags & p) == p) {
                                        return i;
                                    }
                                }
                                throw std::runtime_error("failed to find suitable memory type!");
                            }(memReqs.memoryTypeBits)
                    };
                    VkDeviceMemory memory;
                    if (vkAllocateMemory(dev, &memAllocInfo, nullptr, &memory) != VK_SUCCESS) {
                        throw std::runtime_error("failed to allocate buffer memory!");
                    }
                    vkBindBufferMemory(dev, buffer, memory, 0);
                    return memory;
                }(device, stagingBuffer);
                void* data;
                vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
                memcpy(data, pixels, static_cast<size_t>(imageSize));
                vkUnmapMemory(device, stagingBufferMemory);
                stbi_image_free(pixels);
                auto texImgMemory = [](const auto& dev, const auto& img){
                    VkMemoryRequirements memRequirements;
                    vkGetImageMemoryRequirements(dev, img, &memRequirements);
                    auto imgAllocInfo = VkMemoryAllocateInfo{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                            .allocationSize = memRequirements.size,
                            .memoryTypeIndex = [](const auto& memType){
                                const auto p = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                                VkPhysicalDeviceMemoryProperties mP;
                                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mP);
                                for (uint32_t i = 0; i < mP.memoryTypeCount; i++) {
                                    if ((memType & (1 << i)) && (mP.memoryTypes[i].propertyFlags & p) == p) { return i; }
                                } throw std::runtime_error("failed to find suitable memory type!"); 
                            }(memRequirements.memoryTypeBits),
                    };
                    VkDeviceMemory imageMemory;
                    if (vkAllocateMemory(dev, &imgAllocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
                        throw std::runtime_error("failed to allocate image memory!");
                    }
                    vkBindImageMemory(dev, img, imageMemory, 0);
                    return imageMemory;
                }(device, imgViewInfo.image);
                VkCommandBuffer transferCommandBuffer;
                auto allocInfo = VkCommandBufferAllocateInfo {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                        .commandPool = [](const auto& dev){
                            auto info = VkCommandPoolCreateInfo{
                                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                .queueFamilyIndex = QueueFamilyGST[2]
                            };
                            VkCommandPool cPool;
                            if (vkCreateCommandPool(dev, &info, nullptr, &cPool) != VK_SUCCESS) {
                                throw std::runtime_error("failed to create graphics command pool!");
                            }
                            return cPool;
                        }(device),
                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        .commandBufferCount = 1
                };
                vkAllocateCommandBuffers(device, &allocInfo, &transferCommandBuffer);
                VkCommandBufferBeginInfo beginInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                };
                vkBeginCommandBuffer(transferCommandBuffer, &beginInfo);
                auto tBarrier = VkImageMemoryBarrier {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .srcAccessMask = 0,
                        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = imgViewInfo.image,
                        .subresourceRange = VkImageSubresourceRange{ 
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                        }
                };
                vkCmdPipelineBarrier(transferCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &tBarrier
                        );
                auto copyRegion = VkBufferImageCopy { 
                    .bufferOffset = 0,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                        },
                        .imageOffset = {0, 0, 0 },
                        .imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}
                };
                vkCmdCopyBufferToImage(transferCommandBuffer, stagingBuffer, imgViewInfo.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
                vkEndCommandBuffer(transferCommandBuffer);
                auto submitInfo = VkSubmitInfo {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &transferCommandBuffer
                };
                VkQueue tQueue {};
                vkGetDeviceQueue(device, QueueFamilyGST[2], 0, &tQueue);
                vkQueueSubmit(tQueue, 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(tQueue);
                [](const auto& dev, const auto& image) {
                    const auto writeBarrier = VkImageMemoryBarrier {
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = image,
                            .subresourceRange = VkImageSubresourceRange{ 
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1 }
                    };
                    auto allocInfo = VkCommandBufferAllocateInfo{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                            .commandPool = [](const auto& dev){
                                auto info = VkCommandPoolCreateInfo{
                                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .queueFamilyIndex = QueueFamilyGST[0]
                                };
                                VkCommandPool cPool;
                                if (vkCreateCommandPool(dev, &info, nullptr, &cPool) != VK_SUCCESS) {
                                    throw std::runtime_error("failed to create graphics command pool!");
                                }
                                return cPool;
                            }(dev),
                            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                            .commandBufferCount = 1
                    };
                    VkCommandBuffer commandBuffer;
                    vkAllocateCommandBuffers(dev, &allocInfo, &commandBuffer);
                    VkCommandBufferBeginInfo beginInfo{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                    };
                    vkBeginCommandBuffer(commandBuffer, &beginInfo);
                    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            0,
                            0, nullptr,
                            0, nullptr,
                            1, &writeBarrier
                            );
                    vkEndCommandBuffer(commandBuffer);
                    auto submitInfo = VkSubmitInfo{
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &commandBuffer
                    };
                    VkQueue queue;
                    vkGetDeviceQueue(dev, QueueFamilyGST[0], 0, &queue);
                    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
                    vkQueueWaitIdle(queue);
                    vkFreeCommandBuffers(dev, allocInfo.commandPool, 1, &commandBuffer);
                    vkDestroyCommandPool(dev, allocInfo.commandPool, nullptr);
                }(device, imgViewInfo.image);
                vkFreeCommandBuffers(device, allocInfo.commandPool, 1, &transferCommandBuffer);
                vkDestroyCommandPool(device, allocInfo.commandPool, nullptr);
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingBufferMemory, nullptr);
                return std::pair<VkImageViewCreateInfo, VkDeviceMemory>{imgViewInfo, texImgMemory};
            }(),
                .imageInfo = VkDescriptorImageInfo {
                    .sampler = [](const auto& dev){
                        VkPhysicalDeviceProperties properties{};
                        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
                        auto samplerInfo = VkSamplerCreateInfo {
                            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_LINEAR,
                                .minFilter = VK_FILTER_LINEAR,
                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .anisotropyEnable = VK_TRUE,
                                .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                                .compareEnable = VK_FALSE,
                                .compareOp = VK_COMPARE_OP_ALWAYS,
                                .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                .unnormalizedCoordinates = VK_FALSE
                        };
                        VkSampler sampler;
                        if (vkCreateSampler(dev, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
                            throw std::runtime_error("failed to create texture sampler!");
                        }
                        return sampler;
                    }(device),
                    .imageView = VkImageView{},
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                },
                .uniform = [](const auto& dev){
                    auto ubo = [](){
                        std::array<glm::mat4, 360> u{};
                        std::generate(u.begin(), u.end(), [i = 0]() mutable {
                                return glm::rotate(glm::mat4(1.f), (i++/360.f) * glm::radians(360.f), glm::vec3(0.f, 0.f, 1.f));
                                });
                        return u;
                    }();
                    auto uBufferInfo = VkDescriptorBufferInfo{
                        .buffer = [](){
                            auto info = VkBufferCreateInfo {
                                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size = sizeof(ubo),
                                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
                            };
                            auto uB = VkBuffer{};
                            if (vkCreateBuffer(device, &info, nullptr, &uB) != VK_SUCCESS) {
                                throw std::runtime_error("failed to create buffer!");
                            }
                            return uB;
                        }(),
                            .offset = 0,
                            .range = sizeof(ubo),
                    };
                    VkMemoryRequirements mR;
                    vkGetBufferMemoryRequirements(dev, uBufferInfo.buffer, &mR);
                    auto allocInfo = VkMemoryAllocateInfo {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                            .allocationSize = mR.size,
                            .memoryTypeIndex = [](const auto& memType){
                                const auto p = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                                VkPhysicalDeviceMemoryProperties mP;
                                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mP);
                                for (uint32_t i = 0; i < mP.memoryTypeCount; i++) {
                                    if ((memType & (1 << i)) && (mP.memoryTypes[i].propertyFlags & p) == p) { return i; }
                                } throw std::runtime_error("failed to find suitable memory type!"); 
                            }(mR.memoryTypeBits)
                    };
                    VkDeviceMemory mem;
                    if (vkAllocateMemory(dev, &allocInfo, nullptr, &mem) != VK_SUCCESS) {
                        throw std::runtime_error("failed to allocate buffer memory!");
                    }
                    vkBindBufferMemory(dev, uBufferInfo.buffer, mem, 0);
                    void* data;
                    vkMapMemory(dev, mem, 0, sizeof(ubo), 0, &data);
                    memcpy(data, &ubo[0], sizeof(ubo));
                    vkUnmapMemory(dev, mem);
                    return std::pair<VkDescriptorBufferInfo, VkDeviceMemory>{uBufferInfo, mem};
                }(device),
                .dSetLayout = [](const auto& dev){
                    auto layoutBindings = std::array<VkDescriptorSetLayoutBinding, 2> { VkDescriptorSetLayoutBinding {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                        .pImmutableSamplers = nullptr
                    }, {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        .pImmutableSamplers = nullptr
                    }
                    };
                    auto layoutInfo = VkDescriptorSetLayoutCreateInfo {
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                            .bindingCount = 2,
                            .pBindings = &layoutBindings[0]
                    };
                    VkDescriptorSetLayout layout{};
                    if (vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
                        throw std::runtime_error("failed to create descriptor set layout!");
                    }
                    return layout;
                }(device),
                .allocInfo = VkDescriptorSetAllocateInfo{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = [](const auto& dev){
                        auto poolSizes = std::array<VkDescriptorPoolSize, 2>{VkDescriptorPoolSize {
                            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .descriptorCount = 1
                        }, {
                            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = 1
                        } 
                        };
                        auto poolInfo = VkDescriptorPoolCreateInfo {
                            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                .maxSets = 1,
                                .poolSizeCount = 2,
                                .pPoolSizes = &poolSizes[0],
                        };
                        VkDescriptorPool dPool;
                        if (vkCreateDescriptorPool(dev, &poolInfo, nullptr, &dPool) != VK_SUCCESS) {
                            throw std::runtime_error("failed to create descriptor pool!");
                        }
                        return dPool;
                    }(device),
                    .descriptorSetCount = 1,
                },
                .dSet = VkDescriptorSet{}
        };
        if (vkCreateImageView(device, &descriptor.texture.first, nullptr, &descriptor.imageInfo.imageView) != VK_SUCCESS){
            throw std::runtime_error("failed to create texture image view!");
        }
        descriptor.allocInfo.pSetLayouts = &descriptor.dSetLayout;
        if (vkAllocateDescriptorSets(device, &descriptor.allocInfo, &descriptor.dSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!"); }
        auto write = std::array<VkWriteDescriptorSet, 2> {
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor.dSet,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &descriptor.uniform.first},
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor.dSet,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &descriptor.imageInfo
                }
        };
        vkUpdateDescriptorSets(device, 2, &write[0], 0, nullptr);
        return descriptor; }();
    auto pushConstantRange = VkPushConstantRange {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = 128,
    };
    pipeline.inputBinding = VkVertexInputBindingDescription{
        .binding = 0,
            .stride = sizeof(Pipeline::Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };
    pipeline.attributeDescriptions = std::array<VkVertexInputAttributeDescription, 3> {
        VkVertexInputAttributeDescription{
            .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = 0 }, {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                    .offset = sizeof(glm::vec4) }, {
                        .location = 2,
                        .binding = 0,
                        .format = VK_FORMAT_R32G32_SFLOAT,
                        .offset = offsetof(Pipeline::Vertex, textureCoordinates) } 
    };
    pipeline.vertexInput = VkPipelineVertexInputStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &pipeline.inputBinding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(pipeline.attributeDescriptions.size()),
            .pVertexAttributeDescriptions = pipeline.attributeDescriptions.data()
    };
    auto createShaderModule = [](const auto& dev, const std::string& filename) {
        auto code = [](const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) { throw std::runtime_error("failed to open file!"); }
            size_t fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();
            return buffer;
        }(filename);
        auto info = VkShaderModuleCreateInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };
        VkShaderModule mod;
        if (vkCreateShaderModule(dev, &info, nullptr, &mod) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return mod;
    };
    pipeline.shaderStages =std::vector<VkPipelineShaderStageCreateInfo>{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = createShaderModule(dev, "shaders/vert.spv"),
                .pName = "main"
        }, VkPipelineShaderStageCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = createShaderModule(dev, "shaders/frag.spv"),
                .pName = "main"}
    };
    pipeline.inputAssembly = VkPipelineInputAssemblyStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE};
    pipeline.view = VkViewport {};
    pipeline.scissor = VkRect2D {};
    pipeline.viewportState = VkPipelineViewportStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &pipeline.view,
            .scissorCount = 1,
            .pScissors = &pipeline.scissor
    };
    pipeline.rasterizer = VkPipelineRasterizationStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
    };
    pipeline.multisample = VkPipelineMultisampleStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
    };
    pipeline.colorBlendAttachment = VkPipelineColorBlendAttachmentState {
        .blendEnable = VK_FALSE,
            .colorWriteMask =VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT
    };
    pipeline.depthStencilState = VkPipelineDepthStencilStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
    };
    pipeline.colorBlend = VkPipelineColorBlendStateCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &pipeline.colorBlendAttachment,
            .blendConstants = {}
    };
    pipeline.dyn = std::vector<VkDynamicState>{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    pipeline.dynamicState = VkPipelineDynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates = &pipeline.dyn[0]
    };
    pipeline.attribute = [](){
        const auto verts = [](){
            std::vector<Pipeline::Vertex> vertices(24*12*6);
            auto down = glm::vec4{0.f, 0.0f, 1.f, 1.f};
            for(int i = 0; i < 12; i++){
                auto theta = (i/12.f) * glm::radians(180.f);
                auto theta2= ((i+1)/12.f) * glm::radians(180.f);
                for(int j = 0; j < 24; j++){
                    auto phi = (j/24.f)*glm::radians(360.f);
                    auto phi2= ((j+1)/24.f)*glm::radians(360.f);

                    auto bottomLeft= glm::rotate(phi,  glm::vec3{0.f, 0.f, 1.f})*glm::rotate(theta,  glm::vec3{0.f, 1.f, 0.f})*down;
                    auto bottomRight=glm::rotate(phi2, glm::vec3{0.f, 0.f, 1.f})*glm::rotate(theta,  glm::vec3{0.f, 1.f, 0.f})*down;
                    auto topLeft =   glm::rotate(phi,  glm::vec3{0.f, 0.f, 1.f})*glm::rotate(theta2, glm::vec3{0.f, 1.f, 0.f})*down;
                    auto topRight =  glm::rotate(phi2, glm::vec3{0.f, 0.f, 1.f})*glm::rotate(theta2, glm::vec3{0.f, 1.f, 0.f})*down;
                    vertices[(i*24+j)*6+0] = {bottomLeft, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{(j/24.f), i/12.f}};
                    vertices[(i*24+j)*6+2] = {bottomRight, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{((j+1)/24.f), i/12.f}};
                    vertices[(i*24+j)*6+1] = {topLeft, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{(j/24.f), (i+1)/12.f}};
                    vertices[(i*24+j)*6+3] = {bottomRight, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{((j+1)/24.f), i/12.f}};
                    vertices[(i*24+j)*6+4] = {topLeft, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{(j/24.f), (i+1)/12.f}};
                    vertices[(i*24+j)*6+5] = {topRight, glm::vec4{0.f, 0.f, 1.f, 1.f}, glm::vec2{((j+1)/24.f), (i+1)/12.f}};
                }
            }
            return vertices;
        }();
        const auto inds = [](){
            std::vector<uint16_t> ind(24*12*6);
            std::generate(ind.begin(), ind.end(), [i = 0]() mutable { i++; return i-1; });
            return ind;
        }();
        const auto localMemory = [](const auto& dev, const void* src, const auto& sz, const unsigned int& use) {
            auto Buffer = [](const auto& dev, const auto& size, const uint32_t& use, const auto& queueFamilies){
                auto info = VkBufferCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .size = size,
                        .usage = use,
                        .sharingMode = (queueFamilies.size() >= 2)?VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE,
                        .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
                        .pQueueFamilyIndices = queueFamilies.data()
                };
                VkBuffer b;
                if (vkCreateBuffer(dev, &info, nullptr, &b) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create buffer!");
                }
                return b;
            };
            auto BufferMemory = [](const auto& dev, const auto& buffer, const auto& properties){
                VkMemoryRequirements mR;
                vkGetBufferMemoryRequirements(dev, buffer, &mR);
                auto a = VkMemoryAllocateInfo {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        .allocationSize = mR.size,
                        .memoryTypeIndex = [](const auto& t, const auto& p){
                            VkPhysicalDeviceMemoryProperties mP;
                            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mP);
                            for (uint32_t i = 0; i < mP.memoryTypeCount; i++) {
                                if ((t & (1 << i)) && (mP.memoryTypes[i].propertyFlags & p) == p) {
                                    return i;
                                }
                            }
                            throw std::runtime_error("failed to find suitable memory type!");
                        }(mR.memoryTypeBits, properties)
                };
                VkDeviceMemory mem;
                if (vkAllocateMemory(dev, &a, nullptr, &mem) != VK_SUCCESS) {
                    throw std::runtime_error("failed to allocate buffer memory!");
                }
                vkBindBufferMemory(dev, buffer, mem, 0);
                return mem;
            };
            auto stagingBuffer = Buffer(dev, sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, std::vector<uint32_t>{});
            auto stagingBufferMemory = BufferMemory(dev, stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            void* data;
            vkMapMemory(dev, stagingBufferMemory, 0, sz, 0, &data);
            memcpy(data, src, (size_t) sz);
            vkUnmapMemory(dev, stagingBufferMemory);
            auto buffer = Buffer(dev, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | use, std::vector<uint32_t>{0, 1});
            auto bufferMemory = BufferMemory(dev, buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkCommandBuffer cmdBuf;
            auto allocInfo = VkCommandBufferAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .commandPool = [](const auto& dev){
                        auto info = VkCommandPoolCreateInfo{
                            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                            .queueFamilyIndex = QueueFamilyGST[2]
                        };
                        VkCommandPool cPool;
                        if (vkCreateCommandPool(dev, &info, nullptr, &cPool) != VK_SUCCESS) {
                            throw std::runtime_error("failed to create transfer command pool!");
                        }
                        return cPool;
                    }(dev),
                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    .commandBufferCount = 1
            };
            vkAllocateCommandBuffers(dev, &allocInfo, &cmdBuf);
            auto beginInfo = VkCommandBufferBeginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            };
            vkBeginCommandBuffer(cmdBuf, &beginInfo);
            auto copyRegion = VkBufferCopy { .size = sz };
            vkCmdCopyBuffer(cmdBuf, stagingBuffer, buffer, 1, &copyRegion);
            vkEndCommandBuffer(cmdBuf);
            VkQueue tQueue {};
            vkGetDeviceQueue(dev, QueueFamilyGST[2], 0, &tQueue);
            auto submitInfo = VkSubmitInfo {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &cmdBuf
            };
            vkQueueSubmit(tQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(tQueue);
            vkFreeCommandBuffers(dev, allocInfo.commandPool, 1, &cmdBuf);
            vkDestroyCommandPool(dev, allocInfo.commandPool, nullptr);
            vkDestroyBuffer(dev, stagingBuffer, nullptr);
            vkFreeMemory(dev, stagingBufferMemory, nullptr);
            return std::pair<VkBuffer, VkDeviceMemory>{buffer, bufferMemory};
        };
        return Pipeline::Attribute{
            .vertices=localMemory(device,verts.data(),sizeof(verts[0])*verts.size(),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
                .indices=localMemory(device,inds.data(),sizeof(inds[0])*inds.size(),VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
                .size = static_cast<uint32_t>(inds.size())
        };
    }();
    pipeline.info = VkGraphicsPipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = static_cast<uint32_t>(pipeline.shaderStages.size()),
            .pStages = pipeline.shaderStages.data(),
            .pVertexInputState = &pipeline.vertexInput,
            .pInputAssemblyState = &pipeline.inputAssembly,
            .pViewportState = &pipeline.viewportState,
            .pRasterizationState = &pipeline.rasterizer,
            .pMultisampleState = &pipeline.multisample,
            .pDepthStencilState = &pipeline.depthStencilState,
            .pColorBlendState = &pipeline.colorBlend,
            .pDynamicState = &pipeline.dynamicState,
            .layout = [](const auto& dev, const auto& info){
                VkPipelineLayout layout; 
                if (vkCreatePipelineLayout(dev, &info, nullptr, &layout) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create pipeline layout!");
                }
                return layout;
            }(dev, VkPipelineLayoutCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &pipeline.descriptor.dSetLayout,
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &pushConstantRange
                    }),
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE
    };
    return pipeline;
};
void mainLoop(){
    auto pipe = PrimaryPipe(device);
    swap.format = [&swap](){
        auto a = [](const auto& p, const auto& s){
            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(p, s, &formatCount, nullptr);
            std::vector<VkSurfaceFormatKHR> f(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(p, s, &formatCount, f.data());
            return f;
        }(physicalDevice, swap.info.surface);
        auto res = std::find_if(a.begin(), a.end(), [](const VkSurfaceFormatKHR& f) {
                return f.format==VK_FORMAT_B8G8R8A8_SRGB&&f.colorSpace==VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                });
        return res==a.end()?a[0]:*res;
    }();
    pipe.info.renderPass = [](const auto& dev, const auto& format){
        auto colorAttachment = VkAttachmentDescription {
            .format = swap.format.format,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };
        auto depthAttachment = VkAttachmentDescription {
            .format = VK_FORMAT_D24_UNORM_S8_UINT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        const auto attachs = std::array<VkAttachmentDescription, 2>{colorAttachment, depthAttachment};
        auto colorAttachmentRef = VkAttachmentReference {
            .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        auto depthAttachmentRef = VkAttachmentReference {
            .attachment = 1,
                .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };
        auto subpass = VkSubpassDescription {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentRef,
                .pDepthStencilAttachment = &depthAttachmentRef
        };
        auto dependency = VkSubpassDependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };
        auto info = VkRenderPassCreateInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = 2,
                .pAttachments = &attachs[0],
                .subpassCount = 1,
                .pSubpasses = &subpass,
                .dependencyCount = 1,
                .pDependencies = &dependency};
        VkRenderPass r;
        if (vkCreateRenderPass(dev, &info, nullptr, &r) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
        return r; 
    }(device, swap.format.format);
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe.info, nullptr, &pipe.pipe) != VK_SUCCESS) {
        throw std::runtime_error("graphics pipeline failed");
    }
    vkDestroyShaderModule(device, pipe.shaderStages[0].module, nullptr);
    vkDestroyShaderModule(device, pipe.shaderStages[1].module, nullptr);
    swap.Construct(physicalDevice, device, pipe);
    constexpr int ModularStates = 360;
    std::vector<VkFence> imagesInFlight(swap.images.size(), VK_NULL_HANDLE);
    const auto CommandBuffers = [](const auto& dev, const auto& allocInfo){
        std::vector<VkCommandBuffer> cmds(allocInfo.commandBufferCount);
        if (vkAllocateCommandBuffers(dev, &allocInfo, &cmds[0]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!"); }
        return cmds;
    };
    auto allocInfo2ndary = VkCommandBufferAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = [](const auto& dev){
                auto info = VkCommandPoolCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    .queueFamilyIndex = QueueFamilyGST[0]
                };
                VkCommandPool cmdPool;
                if (vkCreateCommandPool(dev, &info, nullptr, &cmdPool) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create graphics command pool!");
                }
                return cmdPool;
            }(device),
            .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandBufferCount = 1000
    };
    std::vector<std::pair<glm::mat4, uint32_t>> pushes(allocInfo2ndary.commandBufferCount);
    std::generate(pushes.begin(), pushes.end(), [ i = 0 ]() mutable { 
            glm::mat4 proj = glm::perspective(glm::radians(45.f), swap.info.imageExtent.width / (float) swap.info.imageExtent.height, 0.1f, 10.f);
            proj[1][1] *= -1;
            glm::mat4 view = glm::lookAt(glm::vec3(2.f,2.f,2.f),glm::vec3(0.f,0.f,0.f),glm::vec3(0.f,0.f,1.f));
            return std::pair<glm::mat4, uint32_t>{
            proj*view,
            i++%ModularStates,
            };
            });
    auto pushCommand = [&pipe](auto& cmd, auto& push){ 
            auto viewport = VkViewport {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float) swap.info.imageExtent.width,
            .height = (float) swap.info.imageExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0,
            };
            auto scissor = VkRect2D {
            .offset = {0, 0},
            .extent = swap.info.imageExtent
            };
            auto inheritance = VkCommandBufferInheritanceInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            .renderPass = pipe.info.renderPass,
            .subpass = 0,
            .framebuffer = VK_NULL_HANDLE
            };
            auto beginInfo2 = VkCommandBufferBeginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                    .pInheritanceInfo = &inheritance
            };
            if (vkBeginCommandBuffer(cmd, &beginInfo2) != VK_SUCCESS) {
                throw std::runtime_error("command buffer recording failed"); }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            VkDeviceSize offsets = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &pipe.attribute.vertices.first, &offsets);
            vkCmdBindIndexBuffer(cmd, pipe.attribute.indices.first, 0, VK_INDEX_TYPE_UINT16);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.info.layout, 0, 1, &pipe.descriptor.dSet, 0, nullptr);
            vkCmdPushConstants(cmd, pipe.info.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 128, &push);
            vkCmdDrawIndexed(cmd, pipe.attribute.size, 1, 0, 0, 0);
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { throw std::runtime_error("command buffer recording failed"); }
            return cmd; };
    auto allocInfo = VkCommandBufferAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = [](const auto& d){
                auto info = VkCommandPoolCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    .queueFamilyIndex = QueueFamilyGST[0]
                };
                VkCommandPool cmdPool;
                if (vkCreateCommandPool(d, &info, nullptr, &cmdPool) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create graphics command pool!");
                }
                return cmdPool;
            }(device),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(swap.images.size())
    };
    std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers = CommandBuffers(device, allocInfo);
    auto rPassBeginInfo = std::vector<VkRenderPassBeginInfo>{};
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    std::transform(swap.framebuffers.begin(), swap.framebuffers.end(), std::back_inserter(rPassBeginInfo),
            [&clearValues, &pipe](const auto& frame) {
            return VkRenderPassBeginInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pipe.info.renderPass,
            .framebuffer = frame,
            .renderArea = VkRect2D{
            .offset = {0, 0},
            .extent = swap.info.imageExtent },
            .clearValueCount = 2,
            .pClearValues = &clearValues[0],
            }; 
            });
    auto resizeWindow = [&](){
        vkDeviceWaitIdle(device);
        swap.resize(physicalDevice, device, pipe);
        rPassBeginInfo.resize(0);
        std::transform(swap.framebuffers.begin(), swap.framebuffers.end(), std::back_inserter(rPassBeginInfo),
                [&clearValues, &pipe](const auto& frame) {
                return VkRenderPassBeginInfo {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = pipe.info.renderPass,
                .framebuffer = frame,
                .renderArea = VkRect2D{
                .offset = {0, 0},
                .extent = swap.info.imageExtent },
                .clearValueCount = 2,
                .pClearValues = &clearValues[0],
                }; 
                });
        imagesInFlight.resize(swap.images.size(), VK_NULL_HANDLE);
    };
    auto frame = 0;
    auto command2nd = CommandBuffers(device, allocInfo2ndary);
    for(size_t i=0;!glfwWindowShouldClose(swap.window);i=(i+1)%2){
        auto phase = [](){
            static auto startTime = std::chrono::high_resolution_clock::now();
            auto currentTime = std::chrono::high_resolution_clock::now();
            programClock=static_cast<float>(std::chrono::duration<float,std::chrono::seconds::period>(currentTime-startTime).count());
            auto time = static_cast<uint32_t>(programClock);
            return static_cast<uint32_t>(360*programClock);
        }();
        auto commandIndex = frame % command2nd.size();
        auto beginInfo = VkCommandBufferBeginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0
        };
        if((frame++%command2nd.size())==0){
            vkDeviceWaitIdle(device);
            vkFreeCommandBuffers(device, allocInfo2ndary.commandPool, (uint32_t)command2nd.size(), &command2nd[0]);
           command2nd = CommandBuffers(device, allocInfo2ndary);
        }
        glm::mat4 proj=glm::perspective(glm::radians(45.f),swap.info.imageExtent.width/(float) swap.info.imageExtent.height,0.1f,10.f);
        proj[1][1] *= -1;
        glm::mat4 view = glm::lookAt(glm::vec3(2.f,2.f,2.f),glm::vec3(0.f,0.f,0.f),glm::vec3(0.f,0.f,1.f));
        std::pair<glm::mat4, uint32_t> pushVal = std::pair<glm::mat4, uint32_t>{
            proj*view,
                phase%ModularStates,
        };
        command2nd[commandIndex] = pushCommand(command2nd[commandIndex], pushVal);
        vkWaitForFences(device, 1, &syncs[i].f, VK_TRUE, UINT64_MAX);
        uint32_t imgInd;
        VkResult acquired = vkAcquireNextImageKHR(device, swap.swapchain, 1000000000, syncs[i].a, VK_NULL_HANDLE, &imgInd);
        if (acquired == VK_SUBOPTIMAL_KHR || acquired == VK_ERROR_OUT_OF_DATE_KHR ) { resizeWindow(); continue; }
        if (acquired < 0 && acquired != VK_ERROR_OUT_OF_DATE_KHR ) { std::cout << "failed image acquisition\n"; break; } 
        else if (!(acquired==VK_SUCCESS||acquired==VK_SUBOPTIMAL_KHR)) throw std::runtime_error("failed image acquisition");
        glfwPollEvents();
        if (vkBeginCommandBuffer(commandBuffers[imgInd], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!"); }
        vkCmdBeginRenderPass(commandBuffers[imgInd], &rPassBeginInfo[imgInd], VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        vkCmdExecuteCommands(commandBuffers[imgInd], 1, &command2nd[commandIndex]);
        vkCmdEndRenderPass(commandBuffers[imgInd]);
        if (vkEndCommandBuffer(commandBuffers[imgInd]) != VK_SUCCESS) { throw std::runtime_error("failed command recording!"); }
        if (imagesInFlight[imgInd]!=VK_NULL_HANDLE){vkWaitForFences(device,1,&imagesInFlight[imgInd],VK_TRUE,UINT64_MAX);}
        imagesInFlight[imgInd] = syncs[i].f;
        vkResetFences(device, 1, &syncs[i].f);
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        auto submitInfo = VkSubmitInfo {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &syncs[i].a,
                .pWaitDstStageMask = &waitStage,
                .commandBufferCount = 1,
                .pCommandBuffers = &commandBuffers[imgInd],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &syncs[i].b,
        };
        auto presentInfo = VkPresentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &syncs[i].b,
                .swapchainCount = 1,
                .pSwapchains = &swap.swapchain,
                .pImageIndices = &imgInd
        };
        VkQueue graphicsQueue;
        vkGetDeviceQueue(device, QueueFamilyGST[0], 0, &graphicsQueue);
        if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, syncs[i].f) < 0){break;}
        VkQueue presentQueue;
        vkGetDeviceQueue(device, QueueFamilyGST[1], 0, &presentQueue);
        VkResult presented = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) { resizeWindow(); }
    }
    vkDeviceWaitIdle(device);
    for(auto& s: syncs){ s.Destroy(device); }
    vkDestroyImage(device, swap.depthImage, nullptr);
    vkFreeMemory(device, swap.depthImageMemory, nullptr);
    vkDestroyImageView(device, swap.depthImageView, nullptr);
    for (auto frame : swap.framebuffers) { vkDestroyFramebuffer(device, frame, nullptr); }
    for (auto view : swap.imageViews) { vkDestroyImageView(device, view, nullptr); }
    vkDestroySwapchainKHR(device, swap.swapchain, nullptr);
    vkDestroyDescriptorPool(device, pipe.descriptor.allocInfo.descriptorPool, nullptr);
    vkDestroySampler(device, pipe.descriptor.imageInfo.sampler, nullptr);
    vkDestroyImageView(device, pipe.descriptor.imageInfo.imageView, nullptr);
    vkDestroyImage(device, pipe.descriptor.texture.first.image, nullptr);
    vkFreeMemory(device, pipe.descriptor.texture.second, nullptr);
    vkDestroyBuffer(device, pipe.descriptor.uniform.first.buffer, nullptr);
    vkFreeMemory(device, pipe.descriptor.uniform.second, nullptr);
    vkDestroyBuffer(device, pipe.attribute.indices.first, nullptr);
    vkFreeMemory(device, pipe.attribute.indices.second, nullptr);
    vkDestroyBuffer(device, pipe.attribute.vertices.first, nullptr);
    vkFreeMemory(device, pipe.attribute.vertices.second, nullptr);
    vkDestroyPipeline(device, pipe.pipe, nullptr);
    vkDestroyPipelineLayout(device, pipe.info.layout, nullptr);
    vkDestroyRenderPass(device, pipe.info.renderPass, nullptr);
    vkDestroyDescriptorSetLayout(device, pipe.descriptor.dSetLayout, nullptr);
    vkFreeCommandBuffers(device, allocInfo2ndary.commandPool, command2nd.size(), command2nd.data());
    vkFreeCommandBuffers(device, allocInfo.commandPool, commandBuffers.size(), commandBuffers.data());
    vkDestroyCommandPool(device, allocInfo2ndary.commandPool, nullptr);
    vkDestroyCommandPool(device, allocInfo.commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    if (instanceVK.validationLayers.size() >= 1) {
        auto f = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instanceVK.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (f != nullptr) {
            f(instanceVK.instance, instanceVK.debugMessenger, nullptr);
        }
    }
    vkDestroySurfaceKHR(instanceVK.instance, swap.info.surface, nullptr);
    vkDestroyInstance(instanceVK.instance, nullptr);
    glfwDestroyWindow(swap.window);
    glfwTerminate();
};
int main() {
    try {
        if ((instanceVK.validationLayers.size() >= 1) && ![](){
                uint32_t layerCount;
                vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
                std::vector<VkLayerProperties> lP(layerCount);
                vkEnumerateInstanceLayerProperties(&layerCount, lP.data());
                auto v = std::vector<const char*>{ "VK_LAYER_KHRONOS_validation" };
                return std::accumulate(v.begin(),v.end(), true, [&lP]
                        (const bool a, const char* layerName){ 
                        bool layerFound = false;
                        for (const auto& layerProperties : lP) {
                        if (strcmp(layerName, layerProperties.layerName) == 0) {
                        layerFound = true;
                        break;
                        }
                        }
                        return a && layerFound;
                        });
                }()) throw std::runtime_error("validation layers not supported!");
        glfwSetFramebufferSizeCallback(swap.window, framebufferResizeCallback);
        mainLoop();
    } 
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
