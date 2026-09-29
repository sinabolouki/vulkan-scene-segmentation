// httplib must come first (winsock2.h before any windows.h)
#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
#endif
#include "httplib.h"

#include "sam2_client.h"
#include "sam2_io.h"

#include "stb_image.h"   // implementation must be defined in exactly one other .cpp

#include <chrono>
#include <sstream>
#include <iostream>

namespace sam2 {

namespace {

std::string buildSegmentPath(int layer, const SegmentParams& p)
{
    std::ostringstream q;
    q << "/segment?format=png&layer=" << layer;
    if (p.pointsPerSide > 0)          q << "&points_per_side=" << p.pointsPerSide;
    if (p.predIouThresh >= 0.0f)      q << "&pred_iou_thresh=" << p.predIouThresh;
    if (p.stabilityScoreThresh >= 0.0f) q << "&stability_score_thresh=" << p.stabilityScoreThresh;
    return q.str();
}

uint32_t headerU32(const httplib::Response& r, const char* key)
{
    const std::string s = r.get_header_value(key);
    if (s.empty()) return 0;
    try { return static_cast<uint32_t>(std::stoul(s)); } catch (...) { return 0; }
}

float headerF(const httplib::Response& r, const char* key)
{
    const std::string s = r.get_header_value(key);
    if (s.empty()) return 0.0f;
    try { return std::stof(s); } catch (...) { return 0.0f; }
}

} // namespace

bool Sam2Client::health(std::string& outInfo, std::string& outError) const
{
    httplib::Client cli(m_host, m_port);
    cli.set_connection_timeout(m_connectTimeoutSec, 0);
    cli.set_read_timeout(10, 0);

    auto res = cli.Get("/health");
    if (!res)
    {
        outError = "connection failed (" + httplib::to_string(res.error()) + ") to "
                 + m_host + ":" + std::to_string(m_port);
        return false;
    }
    if (res->status != 200)
    {
        outError = "HTTP " + std::to_string(res->status);
        return false;
    }
    outInfo = res->body;
    return true;
}

SegmentResult Sam2Client::segment(const Sam2Renderer::CapturedLayer& layer,
                                  int layerIndex,
                                  const SegmentParams& params) const
{
    SegmentResult out;
    out.layer  = layerIndex;
    out.width  = layer.width;
    out.height = layer.height;

    // ---- 1) Encode layer to PNG in memory --------------------------------
    std::vector<uint8_t> png;
    if (!encodeLayerToPNG(layer, png))
    {
        out.error = "PNG encoding failed";
        return out;
    }
    out.requestBytes = png.size();

    // ---- 2) POST ---------------------------------------------------------
    httplib::Client cli(m_host, m_port);
    cli.set_connection_timeout(m_connectTimeoutSec, 0);
    cli.set_write_timeout(30, 0);
    cli.set_read_timeout(m_readTimeoutSec, 0);

    const auto t0 = std::chrono::steady_clock::now();
    auto res = cli.Post(buildSegmentPath(layerIndex, params),
                        reinterpret_cast<const char*>(png.data()), png.size(),
                        "image/png");
    out.roundTripMs = std::chrono::duration<float, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

    if (!res)
    {
        out.error = "connection failed: " + httplib::to_string(res.error());
        return out;
    }
    if (res->status != 200)
    {
        out.error = "HTTP " + std::to_string(res->status) + ": " + res->body.substr(0, 300);
        return out;
    }

    // ---- 3) Decode 16-bit label map --------------------------------------
    const auto* bytes = reinterpret_cast<const stbi_uc*>(res->body.data());
    const int   len   = static_cast<int>(res->body.size());

    // stbi_load_16 would silently scale an 8-bit PNG by 257 -> check first
    if (!stbi_is_16_bit_from_memory(bytes, len))
    {
        out.error = "response is not a 16-bit PNG label map";
        return out;
    }

    int w = 0, h = 0, comp = 0;
    stbi_us* px = stbi_load_16_from_memory(bytes, len, &w, &h, &comp, 1);
    if (!px)
    {
        out.error = std::string("label map decode failed: ") + stbi_failure_reason();
        return out;
    }
    if (uint32_t(w) != layer.width || uint32_t(h) != layer.height)
    {
        stbi_image_free(px);
        out.error = "label map size mismatch";
        return out;
    }

    out.labels.assign(px, px + size_t(w) * size_t(h));
    stbi_image_free(px);

    // ---- 4) Metadata from headers ----------------------------------------
    out.numMasks      = headerU32(*res, "X-Num-Masks");
    out.inferenceMs   = headerF(*res, "X-Inference-Ms");
    out.serverTotalMs = headerF(*res, "X-Total-Ms");
    out.ok = true;
    return out;
}

} // namespace sam2