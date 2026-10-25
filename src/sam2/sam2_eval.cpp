#include "sam2_eval.h"
#include "sam2_io.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace sam2 {

namespace {

// Running sums so that per-layer scores and the pooled total use the same code.
struct Accumulator
{
    uint64_t pixels = 0, unlabeled = 0;

    uint32_t ufRegions = 0, samMasks = 0;
    double   ufBestSum = 0.0, samBestSum = 0.0;
    double   ufBestWeighted = 0.0, samBestWeighted = 0.0;
    uint64_t ufPixels = 0, samPixels = 0;   // pixels of the regions/masks that were counted

    uint32_t matches50 = 0;
    double   matchedIoUSum = 0.0;

    void add(const Accumulator& o)
    {
        pixels += o.pixels;               unlabeled += o.unlabeled;
        ufRegions += o.ufRegions;         samMasks += o.samMasks;
        ufBestSum += o.ufBestSum;         samBestSum += o.samBestSum;
        ufBestWeighted += o.ufBestWeighted; samBestWeighted += o.samBestWeighted;
        ufPixels += o.ufPixels;           samPixels += o.samPixels;
        matches50 += o.matches50;         matchedIoUSum += o.matchedIoUSum;
    }

    AgreementScore finish() const
    {
        AgreementScore s;
        s.pixels       = pixels;
        s.ufRegions    = ufRegions;
        s.samMasks     = samMasks;
        s.samUnlabeled = pixels    ? float(double(unlabeled) / double(pixels)) : 0.0f;
        s.ufBestIoU    = ufRegions ? float(ufBestSum  / ufRegions) : 0.0f;
        s.samBestIoU   = samMasks  ? float(samBestSum / samMasks)  : 0.0f;
        s.ufCovering   = ufPixels  ? float(ufBestWeighted  / double(ufPixels))  : 0.0f;
        s.samCovering  = samPixels ? float(samBestWeighted / double(samPixels)) : 0.0f;
        s.matches50    = matches50;
        s.matchedIoU   = matches50 ? float(matchedIoUSum / matches50) : 0.0f;
        s.f1At50       = (ufRegions + samMasks)
                       ? float(2.0 * matches50 / double(ufRegions + samMasks)) : 0.0f;
        return s;
    }
};

Accumulator scoreLayer(const std::vector<uint32_t>& regionIds,
                       const std::vector<uint16_t>& samLabels,
                       uint32_t minRegionPixels)
{
    Accumulator acc;

    std::unordered_map<uint32_t, uint32_t> ufSize, samSize;
    std::unordered_map<uint64_t, uint32_t> joint;   // (root << 16) | samLabel -> pixel count

    for (size_t i = 0; i < regionIds.size(); ++i)
    {
        const uint32_t root = regionIds[i];
        if (root == UFRenderer::UF_NULL) continue;   // no fragment at this depth layer

        ++acc.pixels;
        ++ufSize[root];

        const uint16_t mask = samLabels[i];
        if (mask == 0) { ++acc.unlabeled; continue; }

        ++samSize[mask];
        ++joint[(uint64_t(root) << 16) | mask];
    }

    std::unordered_map<uint32_t, float> ufBest, samBest;
    for (const auto& [key, overlap] : joint)
    {
        const uint32_t root = uint32_t(key >> 16);
        const uint16_t mask = uint16_t(key & 0xFFFFu);
        const uint32_t sizeU = ufSize[root];
        const uint32_t sizeS = samSize[mask];
        const float iou = float(overlap) / float(sizeU + sizeS - overlap);

        float& bu = ufBest[root]; if (iou > bu) bu = iou;
        float& bs = samBest[mask]; if (iou > bs) bs = iou;

        // IoU > 0.5 can hold for at most one partner on each side.
        if (iou > 0.5f && sizeU >= minRegionPixels && sizeS >= minRegionPixels)
        {
            ++acc.matches50;
            acc.matchedIoUSum += iou;
        }
    }

    for (const auto& [root, size] : ufSize)
    {
        if (size < minRegionPixels) continue;
        const auto it = ufBest.find(root);
        const float best = (it != ufBest.end()) ? it->second : 0.0f;
        ++acc.ufRegions;
        acc.ufBestSum      += best;
        acc.ufBestWeighted += double(best) * size;
        acc.ufPixels       += size;
    }
    for (const auto& [mask, size] : samSize)
    {
        if (size < minRegionPixels) continue;
        const float best = samBest[mask];   // every SAM2 pixel lies on some UF region
        ++acc.samMasks;
        acc.samBestSum      += best;
        acc.samBestWeighted += double(best) * size;
        acc.samPixels       += size;
    }
    return acc;
}

void writeRow(std::ofstream& out, const std::string& layer, const AgreementScore& s)
{
    out << layer << ',' << s.pixels << ',' << s.ufRegions << ',' << s.samMasks << ','
        << s.samUnlabeled << ',' << s.ufBestIoU << ',' << s.samBestIoU << ','
        << s.ufCovering << ',' << s.samCovering << ',' << s.matches50 << ','
        << s.matchedIoU << ',' << s.f1At50 << '\n';
}

} // namespace

Evaluation evaluateSamVsUnionFind(const LayerSet& layers,
                                  const std::vector<SegmentResult>& results,
                                  uint32_t minRegionPixels)
{
    Evaluation eval;
    eval.minRegionPixels = minRegionPixels;

    Accumulator total;
    const size_t count = std::min(layers.size(), results.size());
    for (size_t i = 0; i < count; ++i)
    {
        const auto& layer  = layers[i];
        const auto& result = results[i];
        const size_t n = size_t(layer.width) * layer.height;

        if (!result.ok || result.skipped) continue;
        if (result.labels.size() != n || layer.regionIds.size() != n) continue;

        const Accumulator acc = scoreLayer(layer.regionIds, result.labels, minRegionPixels);
        total.add(acc);
        eval.layers.push_back({int(i), acc.finish()});
    }
    eval.total = total.finish();
    return eval;
}

bool saveEvaluationCSV(const Evaluation& eval, const std::string& path)
{
    std::error_code ec;
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent, ec);

    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "[sam2_eval] Cannot write " << path << std::endl;
        return false;
    }

    out << "# SAM2 vs union-find, min_region_pixels=" << eval.minRegionPixels << '\n';
    out << "layer,pixels,uf_regions,sam_masks,sam_unlabeled,uf_best_iou,sam_best_iou,"
           "uf_covering,sam_covering,matches_iou50,matched_iou,f1_iou50\n";
    for (const auto& l : eval.layers)
        writeRow(out, std::to_string(l.layer), l.score);
    writeRow(out, "all", eval.total);

    std::cout << "[sam2_eval] Wrote " << path << std::endl;
    return true;
}

size_t saveRegionLabelMaps(const LayerSet& layers, const std::string& directory)
{
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec)
    {
        std::cerr << "[sam2_eval] create_directories failed: " << directory
                  << " - " << ec.message() << std::endl;
        return 0;
    }

    // One root -> color id table for all layers, so a region keeps its color across layers.
    std::unordered_map<uint32_t, uint16_t> compactId;
    size_t written = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        const auto& layer = layers[i];
        const size_t n = size_t(layer.width) * layer.height;
        if (layer.regionIds.size() != n || isLayerEmpty(layer)) continue;

        std::vector<uint16_t> ids(n, 0);
        for (size_t p = 0; p < n; ++p)
        {
            const uint32_t root = layer.regionIds[p];
            if (root == UFRenderer::UF_NULL) continue;
            auto [it, inserted] = compactId.try_emplace(root, uint16_t(0));
            if (inserted) it->second = uint16_t((compactId.size() - 1) % 65535 + 1);   // 0 = no region
            ids[p] = it->second;
        }

        char name[64];
        std::snprintf(name, sizeof(name), "layer_%02zu_uf.png", i);
        if (saveLabelMapColorized(ids, layer.width, layer.height, (fs::path(directory) / name).string()))
            ++written;
    }
    return written;
}

} // namespace sam2
