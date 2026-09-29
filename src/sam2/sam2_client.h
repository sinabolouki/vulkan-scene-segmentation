#pragma once

// NOTE: do NOT include httplib.h here (winsock/windows.h include-order issues).

#include "sam2_renderer.h"
#include <string>
#include <vector>
#include <cstdint>

namespace sam2 {

struct SegmentParams
{
    int   pointsPerSide        = 0;      // 0   = server default
    float predIouThresh        = -1.0f;  // < 0 = server default
    float stabilityScoreThresh = -1.0f;  // < 0 = server default
};

struct SegmentResult
{
    bool        ok      = false;
    bool        skipped = false;   // empty layer, not sent
    std::string error;

    int      layer    = -1;
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t numMasks = 0;

    float  inferenceMs   = 0.0f;   // SAM2 only (server-side, CUDA-synchronized)
    float  serverTotalMs = 0.0f;   // server incl. filtering / label map
    float  roundTripMs   = 0.0f;   // client-side: request -> response
    size_t requestBytes  = 0;

    std::vector<uint16_t> labels;  // width*height, row-major, 0 = no mask
};

class Sam2Client
{
public:
    Sam2Client() = default;
    Sam2Client(std::string host, int port) : m_host(std::move(host)), m_port(port) {}

    void setEndpoint(const std::string& host, int port) { m_host = host; m_port = port; }
    void setTimeouts(int connectSec, int readSec) { m_connectTimeoutSec = connectSec; m_readTimeoutSec = readSec; }

    const std::string& host() const { return m_host; }
    int                port() const { return m_port; }

    // GET /health. outInfo = raw JSON body on success.
    bool health(std::string& outInfo, std::string& outError) const;

    // POST /segment?format=png. Blocking. Thread-safe (no shared state).
    SegmentResult segment(const Sam2Renderer::CapturedLayer& layer,
                          int layerIndex,
                          const SegmentParams& params = {}) const;

private:
    std::string m_host = "127.0.0.1";
    int         m_port = 8765;
    int         m_connectTimeoutSec = 3;
    int         m_readTimeoutSec    = 600;   // SAM2 can be slow for dense layers
};

} // namespace sam2