#pragma once

// CPU-only, header-only helpers. No Vulkan or GLM dependencies.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace uf_eval {

inline constexpr std::uint32_t null_index = 0xFFFFFFFFu;

struct Counts {
    std::uint64_t visible = 0;
    std::uint64_t occluded = 0;
    std::uint64_t total() const { return visible + occluded; }
};

struct Analysis {
    // High 32 bits: canonical root (null_index means invalid).
    // Low 32 bits: reference objectID, including objectID zero.
    std::unordered_map<std::uint64_t, Counts> overlaps;
    std::uint64_t retained = 0;
    std::uint64_t visible = 0;
    std::uint64_t occluded = 0;
    std::uint64_t invalid_root_fragments = 0;
    std::uint64_t nonflat_parent_fragments = 0;
};

struct Metadata {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t node_capacity = 0;
    std::uint32_t allocation_requests = 0;
    std::uint32_t per_pixel_limit = 32;
    std::uint32_t merge_iterations = 2;
    std::uint32_t merge_mode = 0;
    float proximity_scale = 0;
    float plane_threshold = 0;
    float color_threshold = 0;
};

// Node must provide uint32_t-compatible objectID and next members.
// nodes and parents contain the successfully allocated prefix of one snapshot.
template<class Node>
Analysis analyze(
    std::span<const Node> nodes,
    std::span<const std::uint32_t> parents,
    std::span<const std::uint32_t> heads,
    std::uint32_t per_pixel_limit)
{
    if (nodes.size() != parents.size() || nodes.size() >= null_index)
        throw std::runtime_error("Invalid node/parent snapshot sizes");
    if (per_pixel_limit == 0)
        throw std::runtime_error("Per-pixel limit must be positive");

    Analysis result;
    std::vector<std::uint8_t> retained(nodes.size(), 0);

    // Enumerate reachable nodes, never all allocation slots: sorting may have
    // made some successfully allocated nodes unreachable.
    for (std::uint32_t head : heads) {
        std::uint32_t index = head;
        std::uint32_t rank = 0;
        while (index != null_index) {
            if (index >= nodes.size())
                throw std::runtime_error("List contains an out-of-range node");
            if (rank >= per_pixel_limit)
                throw std::runtime_error("List exceeds its retention bound");
            if (retained[index])
                throw std::runtime_error("Repeated node or cycle in fragment lists");
            retained[index] = 1;
            ++result.retained;
            if (rank == 0) ++result.visible;
            else ++result.occluded;
            ++rank;
            index = nodes[index].next;
        }
    }

    // State 0: unseen; 1: on the current traversal; 2: cached result.
    // Full CPU root resolution avoids interpreting an intermediate parent as a
    // distinct region. Invalid roots remain represented in the output.
    std::vector<std::uint8_t> state(nodes.size(), 0);
    std::vector<std::uint32_t> roots(nodes.size(), null_index);
    std::vector<std::uint32_t> path;
    const auto resolve = [&](std::uint32_t start) -> std::uint32_t {
        path.clear();
        std::uint32_t current = start;
        std::uint32_t root = null_index;
        for (;;) {
            if (current >= nodes.size() || !retained[current]) break;
            if (state[current] == 2) {
                root = roots[current];
                break;
            }
            if (state[current] == 1) break; // parent cycle
            state[current] = 1;
            path.push_back(current);
            const std::uint32_t parent = parents[current];
            if (parent == current) {
                root = current;
                break;
            }
            current = parent;
        }
        for (std::uint32_t index : path) {
            roots[index] = root;
            state[index] = 2;
        }
        return root;
    };

    for (std::uint32_t head : heads) {
        std::uint32_t index = head;
        std::uint32_t rank = 0;
        while (index != null_index) {
            const std::uint32_t root = resolve(index);
            if (root == null_index) ++result.invalid_root_fragments;
            else if (parents[index] != root) ++result.nonflat_parent_fragments;
            const auto key =
                (std::uint64_t(root) << 32) | std::uint64_t(nodes[index].objectID);
            auto& counts = result.overlaps[key];
            if (rank == 0) ++counts.visible;
            else ++counts.occluded;
            ++rank;
            index = nodes[index].next;
        }
    }
    return result;
}

inline void write_csv(
    const std::filesystem::path& destination,
    const Analysis& analysis,
    const Metadata& metadata)
{
    if (destination.empty())
        throw std::runtime_error("Empty CSV filename");
    if (std::filesystem::exists(destination))
        throw std::runtime_error("CSV already exists; choose a new filename");
    if (!destination.parent_path().empty())
        std::filesystem::create_directories(destination.parent_path());

    auto temporary = destination;
    temporary += ".tmp";
    if (std::filesystem::exists(temporary))
        throw std::runtime_error("Temporary CSV already exists");

    const auto stored = std::min(
        metadata.allocation_requests, metadata.node_capacity);
    if (analysis.retained > stored)
        throw std::runtime_error("Retained count exceeds successfully stored nodes");

    try {
        std::ofstream stream;
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream.open(temporary, std::ios::out | std::ios::trunc);
        stream.imbue(std::locale::classic());
        stream << std::setprecision(std::numeric_limits<float>::max_digits10);
        stream << "# format=uf_overlap_v1\n"
               << "# width=" << metadata.width << '\n'
               << "# height=" << metadata.height << '\n'
               << "# node_capacity=" << metadata.node_capacity << '\n'
               << "# allocation_requests=" << metadata.allocation_requests << '\n'
               << "# stored_fragments=" << stored << '\n'
               << "# retained_fragments=" << analysis.retained << '\n'
               << "# unretained_stored_fragments=" << stored - analysis.retained << '\n'
               << "# allocation_overflow="
               << (metadata.allocation_requests > metadata.node_capacity ? 1 : 0) << '\n'
               << "# per_pixel_limit=" << metadata.per_pixel_limit << '\n'
               << "# merge_iterations=" << metadata.merge_iterations << '\n'
               << "# merge_mode=" << metadata.merge_mode << '\n'
               << "# proximity_scale=" << metadata.proximity_scale << '\n'
               << "# plane_threshold=" << metadata.plane_threshold << '\n'
               << "# color_threshold=" << metadata.color_threshold << '\n'
               << "# invalid_root_fragments=" << analysis.invalid_root_fragments << '\n'
               << "# nonflat_parent_fragments=" << analysis.nonflat_parent_fragments << '\n'
               << "# roots_resolved_on_cpu=1\n"
               << "region_id,object_id,total_count,visible_count,occluded_count\n";

        std::vector<std::uint64_t> keys;
        keys.reserve(analysis.overlaps.size());
        for (const auto& entry : analysis.overlaps) keys.push_back(entry.first);
        std::sort(keys.begin(), keys.end());
        for (std::uint64_t key : keys) {
            const auto root = std::uint32_t(key >> 32);
            const auto object = std::uint32_t(key);
            const auto& counts = analysis.overlaps.at(key);
            if (root == null_index) stream << "-1";
            else stream << root;
            stream << ',' << object << ',' << counts.total()
                   << ',' << counts.visible << ',' << counts.occluded << '\n';
        }
        stream.close();
        std::filesystem::rename(temporary, destination);
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace uf_eval
