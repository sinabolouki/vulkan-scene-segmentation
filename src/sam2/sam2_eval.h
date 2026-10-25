#pragma once

#include "sam2_job.h"   // LayerSet, SegmentResult

#include <string>
#include <vector>
#include <cstdint>

namespace sam2 {

// Agreement between SAM2 masks and union-find regions on one depth layer (or on all
// evaluated layers pooled). Only pixels with a fragment at that depth layer count.
// Regions/masks smaller than minRegionPixels are left out of the per-region averages
// and match counts, but their pixels still count towards the other side's IoU.
struct AgreementScore
{
    uint64_t pixels       = 0;     // pixels with a fragment at this depth layer
    uint32_t ufRegions    = 0;     // union-find regions >= minRegionPixels
    uint32_t samMasks     = 0;     // SAM2 masks >= minRegionPixels
    float    samUnlabeled = 0.0f;  // fraction of pixels SAM2 left without a mask

    float ufBestIoU   = 0.0f;      // mean over UF regions of the best IoU with any SAM2 mask
    float samBestIoU  = 0.0f;      // mean over SAM2 masks of the best IoU with any UF region
    float ufCovering  = 0.0f;      // ufBestIoU weighted by region size
    float samCovering = 0.0f;      // samBestIoU weighted by mask size

    uint32_t matches50  = 0;       // region/mask pairs with IoU > 0.5 (one-to-one by construction)
    float    matchedIoU = 0.0f;    // mean IoU of those pairs
    float    f1At50     = 0.0f;    // 2 * matches50 / (ufRegions + samMasks)
};

struct LayerAgreement
{
    int            layer = -1;
    AgreementScore score;
};

struct Evaluation
{
    uint32_t                    minRegionPixels = 0;
    std::vector<LayerAgreement> layers;   // only layers with a successful SAM2 result
    AgreementScore              total;    // all evaluated layers pooled
};

// Compares results[i].labels (SAM2, 0 = no mask) with layers[i].regionIds
// (union-find root, UF_NULL = no fragment) for every layer SAM2 segmented.
Evaluation evaluateSamVsUnionFind(const LayerSet& layers,
                                  const std::vector<SegmentResult>& results,
                                  uint32_t minRegionPixels);

bool saveEvaluationCSV(const Evaluation& eval, const std::string& path);

// Writes layer_XX_uf.png (colorized union-find regions) for every non-empty layer.
// Returns number of files written.
size_t saveRegionLabelMaps(const LayerSet& layers, const std::string& directory);

} // namespace sam2
