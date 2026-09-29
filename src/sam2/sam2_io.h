#pragma once

#include "sam2_renderer.h"
#include <string>
#include <vector>
#include <cstdint>

namespace sam2 {

// ---- Captured layers ------------------------------------------------------

// True if no pixel has alpha > 0 (no fragment at this depth layer).
bool isLayerEmpty(const Sam2Renderer::CapturedLayer& layer);

// RGBA8 layer -> PNG bytes in memory (for HTTP upload).
bool encodeLayerToPNG(const Sam2Renderer::CapturedLayer& layer, std::vector<uint8_t>& outPng);

bool saveLayerToPNG(const Sam2Renderer::CapturedLayer& layer, const std::string& path);

// Writes layer_00.png, layer_01.png, ... Returns number of files written.
size_t saveCapturedLayers(const std::vector<Sam2Renderer::CapturedLayer>& layers,
                          const std::string& directory,
                          bool skipEmpty = true);

// ---- Label maps (SAM2 results) --------------------------------------------

// Deterministic pseudo-random color per mask id (id 0 = black).
void labelColor(uint16_t id, uint8_t outRgb[3]);

// 16-bit label map -> colored RGB PNG (viewable).
bool saveLabelMapColorized(const std::vector<uint16_t>& labels,
                           uint32_t width, uint32_t height,
                           const std::string& path);

// Layer image blended with mask colors. Empty pixels (alpha 0) stay black.
bool saveLabelOverlay(const Sam2Renderer::CapturedLayer& layer,
                      const std::vector<uint16_t>& labels,
                      const std::string& path,
                      float blend = 0.45f);

} // namespace sam2