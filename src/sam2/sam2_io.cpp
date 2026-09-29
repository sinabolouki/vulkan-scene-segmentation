#include "sam2_io.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <filesystem>
#include <iostream>
#include <cstdio>

namespace fs = std::filesystem;

namespace sam2 {

// ─────────────────────────────────────────────────────────────────────────────
// Captured layers
// ─────────────────────────────────────────────────────────────────────────────

bool isLayerEmpty(const Sam2Renderer::CapturedLayer& layer)
{
    const size_t n = size_t(layer.width) * layer.height;
    if (layer.rgba.size() < n * 4) return true;
    for (size_t i = 0; i < n; ++i)
        if (layer.rgba[i * 4 + 3] != 0) return false;
    return true;
}

static void appendToVector(void* context, void* data, int size)
{
    auto* out = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

bool encodeLayerToPNG(const Sam2Renderer::CapturedLayer& layer, std::vector<uint8_t>& outPng)
{
    outPng.clear();
    if (layer.rgba.empty() || layer.width == 0 || layer.height == 0) return false;

    outPng.reserve(size_t(layer.width) * layer.height); // rough guess, avoids many reallocs
    const int ok = stbi_write_png_to_func(appendToVector, &outPng,
                                          int(layer.width), int(layer.height), 4,
                                          layer.rgba.data(), int(layer.width) * 4);
    return ok != 0 && !outPng.empty();
}

bool saveLayerToPNG(const Sam2Renderer::CapturedLayer& layer, const std::string& path)
{
    if (layer.rgba.empty() || layer.width == 0 || layer.height == 0)
    {
        std::cerr << "[sam2_io] Empty layer, skipping: " << path << std::endl;
        return false;
    }
    if (!stbi_write_png(path.c_str(), int(layer.width), int(layer.height), 4,
                        layer.rgba.data(), int(layer.width) * 4))
    {
        std::cerr << "[sam2_io] stbi_write_png failed: " << path << std::endl;
        return false;
    }
    std::cout << "[sam2_io] Wrote " << path << std::endl;
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
        std::cerr << "[sam2_io] create_directories failed: " << directory
                  << " - " << ec.message() << std::endl;
        return 0;
    }

    size_t written = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (skipEmpty && isLayerEmpty(layers[i])) continue;

        char name[64];
        std::snprintf(name, sizeof(name), "layer_%02zu.png", i);
        if (saveLayerToPNG(layers[i], (fs::path(directory) / name).string()))
            ++written;
    }
    std::cout << "[sam2_io] Total written: " << written << " / " << layers.size() << std::endl;
    return written;
}

// ─────────────────────────────────────────────────────────────────────────────
// Label maps
// ─────────────────────────────────────────────────────────────────────────────

void labelColor(uint16_t id, uint8_t outRgb[3])
{
    if (id == 0) { outRgb[0] = outRgb[1] = outRgb[2] = 0; return; }

    uint32_t x = uint32_t(id) * 2654435761u;   // integer hash -> stable colors across runs
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;

    outRgb[0] = uint8_t(64 + ( x        & 0xFF) % 192);
    outRgb[1] = uint8_t(64 + ((x >>  8) & 0xFF) % 192);
    outRgb[2] = uint8_t(64 + ((x >> 16) & 0xFF) % 192);
}

bool saveLabelMapColorized(const std::vector<uint16_t>& labels,
                           uint32_t width, uint32_t height,
                           const std::string& path)
{
    const size_t n = size_t(width) * height;
    if (labels.size() != n || n == 0) return false;

    std::vector<uint8_t> rgb(n * 3);
    for (size_t i = 0; i < n; ++i)
        labelColor(labels[i], &rgb[i * 3]);

    return stbi_write_png(path.c_str(), int(width), int(height), 3, rgb.data(), int(width) * 3) != 0;
}

bool saveLabelOverlay(const Sam2Renderer::CapturedLayer& layer,
                      const std::vector<uint16_t>& labels,
                      const std::string& path,
                      float blend)
{
    const size_t n = size_t(layer.width) * layer.height;
    if (labels.size() != n || layer.rgba.size() < n * 4 || n == 0) return false;

    std::vector<uint8_t> rgb(n * 3);
    for (size_t i = 0; i < n; ++i)
    {
        const uint8_t* src = &layer.rgba[i * 4];
        uint8_t*       dst = &rgb[i * 3];

        if (src[3] == 0) { dst[0] = dst[1] = dst[2] = 0; continue; }  // no fragment here

        if (labels[i] == 0)
        {
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            continue;
        }

        uint8_t c[3];
        labelColor(labels[i], c);
        for (int k = 0; k < 3; ++k)
            dst[k] = uint8_t((1.0f - blend) * src[k] + blend * c[k] + 0.5f);
    }
    return stbi_write_png(path.c_str(), int(layer.width), int(layer.height), 3,
                          rgb.data(), int(layer.width) * 3) != 0;
}

} // namespace sam2