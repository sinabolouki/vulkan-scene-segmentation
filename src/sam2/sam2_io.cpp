#include "sam2_io.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <filesystem>
#include <iostream>
#include <cstdio>

namespace fs = std::filesystem;

namespace sam2 {

bool saveLayerToPNG(const Sam2Renderer::CapturedLayer& layer,
                    const std::string& path)
{
    if (layer.rgba.empty() || layer.width == 0 || layer.height == 0)
    {
        std::cerr << "[sam2_io] Empty layer, skipping: " << path << std::endl;
        return false;
    }

    const int stride = int(layer.width) * 4;
    int ok = stbi_write_png(path.c_str(),
                            int(layer.width),
                            int(layer.height),
                            4,                 // RGBA channels
                            layer.rgba.data(),
                            stride);
    if (!ok)
    {
        std::cerr << "[sam2_io] stbi_write_png failed: " << path << std::endl;
        return false;
    }

    std::cout << "[sam2_io] Wrote " << path
              << " (" << layer.width << "x" << layer.height << ")" << std::endl;
    return true;
}

static bool isEmptyLayer(const Sam2Renderer::CapturedLayer& layer)
{
    // Fast scan: consider empty if all alpha bytes are 0.
    // (Our resolve outputs alpha=1.0 for real fragments, 0 for empty.)
    const size_t n = size_t(layer.width) * layer.height;
    for (size_t i = 0; i < n; ++i)
    {
        if (layer.rgba[i * 4 + 3] != 0) return false;
    }
    return true;
}

size_t saveCapturedLayers(const std::vector<Sam2Renderer::CapturedLayer>& layers,
                          const std::string& directory,
                          bool skipEmpty)
{
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec)
    {
        std::cerr << "[sam2_io] create_directories failed: "
                  << directory << " - " << ec.message() << std::endl;
        return 0;
    }

    size_t written = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (skipEmpty && isEmptyLayer(layers[i]))
        {
            std::cout << "[sam2_io] Layer " << i << " empty, skipped." << std::endl;
            continue;
        }

        char name[64];
        std::snprintf(name, sizeof(name), "layer_%02zu.png", i);
        const std::string path = (fs::path(directory) / name).string();

        if (saveLayerToPNG(layers[i], path))
            ++written;
    }

    std::cout << "[sam2_io] Total written: " << written
              << " / " << layers.size() << std::endl;
    return written;
}

} // namespace sam2