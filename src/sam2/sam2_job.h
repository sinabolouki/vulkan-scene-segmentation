#pragma once

#include "sam2_client.h"
#include "sam2_renderer.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sam2 {

using LayerSet = std::vector<Sam2Renderer::CapturedLayer>;

struct JobConfig
{
    std::string   host = "127.0.0.1";
    int           port = 8765;
    SegmentParams params;
    std::string   maskDir;            // empty = don't write mask PNGs
    bool          saveImages = true;
};

enum class JobState { Idle, Running, Finished, Cancelled, Failed };

struct JobProgress
{
    JobState    state        = JobState::Idle;
    uint32_t    total        = 0;     // non-empty layers to segment
    uint32_t    done         = 0;     // non-empty layers processed (ok or failed)
    int         currentLayer = -1;    // layer currently at the server, -1 = none
    float       elapsedSec   = 0.0f;
    std::string message;
};

/**
 * Runs SAM2 segmentation for a set of captured layers on a worker thread.
 *
 * Threading contract:
 *  - start/cancel/takeResults/progress/join are called from the MAIN thread.
 *  - The worker never touches Vulkan or ImGui.
 *  - Exactly one SegmentResult is delivered per layer (ok, skipped, failed or cancelled).
 */
class Sam2Job
{
public:
    Sam2Job() = default;
    ~Sam2Job();

    Sam2Job(const Sam2Job&)            = delete;
    Sam2Job& operator=(const Sam2Job&) = delete;

    // Returns false if a job is already running or there is nothing to do.
    bool start(std::shared_ptr<const LayerSet> layers, const JobConfig& cfg);

    // Request cancellation. Takes effect after the current layer.
    void cancel();

    bool isRunning() const { return m_running.load(); }

    // Moves all results finished since the last call out of the queue.
    std::vector<SegmentResult> takeResults();

    JobProgress progress() const;

    // Blocks until the worker thread has exited.
    void join();

private:
    void workerMain();
    void pushResult(SegmentResult&& r);
    void setCurrentLayer(int layer);
    void finish(JobState state, const std::string& message);

    std::thread m_thread;

    // Owned by the worker while it runs (written only in start()).
    std::shared_ptr<const LayerSet>       m_layers;
    JobConfig                             m_cfg;
    std::chrono::steady_clock::time_point m_startTime;

    std::atomic<bool> m_cancel{false};
    std::atomic<bool> m_running{false};

    // Guarded by m_mutex
    mutable std::mutex         m_mutex;
    std::vector<SegmentResult> m_pending;
    JobProgress                m_progress;
};

} // namespace sam2