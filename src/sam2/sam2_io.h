#pragma once

#include "sam2_renderer.h"
#include <string>
#include <vector>

namespace sam2 {

// Save a single captured layer to a PNG file.
// Returns true on success.
bool saveLayerToPNG(const Sam2Renderer::CapturedLayer& layer,
                    const std::string& path);

// Save an entire capture batch to a directory as layer_00.png, layer_01.png, ...
// Creates the directory if it doesn't exist.
// If skipEmpty=true, layers where all pixels are (0,0,0,0) are skipped.
// Returns the number of files actually written.
size_t saveCapturedLayers(const std::vector<Sam2Renderer::CapturedLayer>& layers,
                          const std::string& directory,
                          bool skipEmpty = true);

} // namespace sam2