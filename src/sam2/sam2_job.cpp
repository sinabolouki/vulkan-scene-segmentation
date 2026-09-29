#include "sam2_job.h"
#include "sam2_io.h"

#include <cstdio>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace sam2 {

namespace {
float secondsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Main-thread API
// ─────────────────────────────────────────────────────────────────────────────

Sam2Job::~Sam2Job()
{
    cancel();
    join();
}

bool Sam2Job::start(std::shared_ptr<const LayerSet> layers, const JobConfig& cfg)
{
    if (m_running.load())           return false;
    if (!layers || layers->empty()) return false;

    join();   // reap the previous (finished) worker thread, if any

    m_layers    = std::move(layers);
    m_cfg       = cfg;
    m_startTime = std::chrono::steady_clock::now();
    m_cancel.store(false);

    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_pending.clear();
        m_progress         = JobProgress{};
        m_progress.state   = JobState::Running;
        m_progress.message = "Starting...";
    }

    m_running.store(true);
    // Thread creation synchronizes: the worker sees all member writes above.
    m_thread = std::thread(&Sam2Job::workerMain, this);
    return true;
}

void Sam2Job::cancel()
{
    m_cancel.store(true);
}

void Sam2Job::join()
{
    if (m_thread.joinable())
        m_thread.join();
}

std::vector<SegmentResult> Sam2Job::takeResults()
{
    std::vector<SegmentResult> out;
    std::lock_guard<std::mutex> lk(m_mutex);
    out.swap(m_pending);
    return out;
}

JobProgress Sam2Job::progress() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    JobProgress p = m_progress;
    if (p.state == JobState::Running)
        p.elapsedSec = secondsSince(m_startTime);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker-side helpers
// ─────────────────────────────────────────────────────────────────────────────

void Sam2Job::pushResult(SegmentResult&& r)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_pending.push_back(std::move(r));
}

void Sam2Job::setCurrentLayer(int layer)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_progress.currentLayer = layer;
}

void Sam2Job::finish(JobState state, const std::string& message)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_progress.state        = state;
        m_progress.message      = message;
        m_progress.currentLayer = -1;
        m_progress.elapsedSec   = secondsSince(m_startTime);
    }
    // Must be the LAST thing the worker does: all results are already queued.
    m_running.store(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker thread
// ─────────────────────────────────────────────────────────────────────────────

void Sam2Job::workerMain()
{
    const LayerSet& layers = *m_layers;
    const size_t    n      = layers.size();

    auto makeStub = [&](size_t i, const char* error)
    {
        SegmentResult r;
        r.layer  = int(i);
        r.width  = layers[i].width;
        r.height = layers[i].height;
        r.error  = error;
        return r;
    };

    Sam2Client client(m_cfg.host, m_cfg.port);

    // ---- 1) Fail fast if the server is down ------------------------------
    {
        std::string info, err;
        if (!client.health(info, err))
        {
            for (size_t i = 0; i < n; ++i)
                pushResult(makeStub(i, "not processed (server not reachable)"));
            finish(JobState::Failed, "SAM2 server not reachable: " + err);
            return;
        }
    }

    // ---- 2) Find the layers that need work -------------------------------
    std::vector<bool> empty(n, false);
    uint32_t total = 0;
    for (size_t i = 0; i < n; ++i)
    {
        empty[i] = isLayerEmpty(layers[i]);
        if (!empty[i]) ++total;
    }
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_progress.total   = total;
        m_progress.message = "Segmenting...";
    }

    bool     save = m_cfg.saveImages && !m_cfg.maskDir.empty();
    fs::path maskDir;
    if (save)
    {
        maskDir = m_cfg.maskDir;
        std::error_code ec;
        fs::create_directories(maskDir, ec);
        if (ec)
        {
            std::cerr << "[Sam2Job] Cannot create " << maskDir.string()
                      << ": " << ec.message() << " -> mask images disabled" << std::endl;
            save = false;
        }
    }

    // ---- 3) Segment layer by layer ---------------------------------------
    uint32_t done = 0, failed = 0, totalMasks = 0;
    bool     serverLost = false;

    for (size_t i = 0; i < n; ++i)
    {
        if (empty[i])
        {
            SegmentResult r = makeStub(i, "");
            r.ok      = true;
            r.skipped = true;
            pushResult(std::move(r));
            continue;
        }
        if (m_cancel.load())
        {
            pushResult(makeStub(i, "not processed (cancelled)"));
            continue;
        }
        if (serverLost)
        {
            pushResult(makeStub(i, "not processed (server lost)"));
            continue;
        }

        setCurrentLayer(int(i));
        SegmentResult r = client.segment(layers[i], int(i), m_cfg.params);   // blocking HTTP

        if (r.ok)
        {
            totalMasks += r.numMasks;
            if (save)
            {
                char name[64];
                std::snprintf(name, sizeof(name), "layer_%02zu_labels_color.png", i);
                saveLabelMapColorized(r.labels, r.width, r.height, (maskDir / name).string());
                std::snprintf(name, sizeof(name), "layer_%02zu_overlay.png", i);
                saveLabelOverlay(layers[i], r.labels, (maskDir / name).string());
            }
            std::cout << "[Sam2Job] Layer " << i << ": " << r.numMasks << " masks, SAM2 "
                      << r.inferenceMs << " ms, round trip " << r.roundTripMs << " ms" << std::endl;
        }
        else
        {
            ++failed;
            std::cerr << "[Sam2Job] Layer " << i << " FAILED: " << r.error << std::endl;
            // Server died mid-run -> don't wait for timeouts on every remaining layer
            if (r.error.rfind("connection failed", 0) == 0)
                serverLost = true;
        }

        ++done;
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_progress.done         = done;
            m_progress.currentLayer = -1;
        }
        pushResult(std::move(r));   // move AFTER saving the images
    }

    // ---- 4) Final state ----------------------------------------------------
    const bool cancelled = m_cancel.load();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s: %u / %u layer(s) segmented, %u failed, %u masks total, %.1f s",
                  cancelled ? "Cancelled" : (serverLost ? "Aborted (server lost)" : "Done"),
                  done - failed, total, failed, totalMasks, secondsSince(m_startTime));

    std::cout << "[Sam2Job] " << buf << std::endl;
    finish(cancelled ? JobState::Cancelled
                     : (serverLost ? JobState::Failed : JobState::Finished),
           buf);
}

} // namespace sam2