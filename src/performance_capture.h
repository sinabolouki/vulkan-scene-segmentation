#pragma once

// Header-only: the uploaded source archive has no CMakeLists.txt. No new .cpp
// registration is needed. This code has no Vulkan, GLFW, or ImGui dependency.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace performance {
using Clock = std::chrono::steady_clock;
using Metadata = std::vector<std::pair<std::string, std::string>>;

struct GpuFrameResult {
    uint64_t frameId = 0; // Application frame ID, never the reusable flight slot.
    std::optional<double> groupingMs, rendererMs;
    std::optional<double> clearMs, collectMs, sortMs, initParentsMs;
    std::optional<double> extractionMs, compositeMs;
    std::optional<uint64_t> allocationRequests, storedFragments;
    std::optional<uint64_t> retainedFragments, nodePoolOverflow;
    std::optional<uint64_t> perPixelRetentionLosses;
    std::string status = "not_submitted";
};

struct Sample {
    uint64_t frameId = 0;
    Clock::time_point start{};
    std::optional<double> frameMs;
    GpuFrameResult gpu;
    bool gpuResolved = false;
};

inline std::string csvCell(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result = "\"";
    for (char c : value) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}

template<class T> inline std::string number(T value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << value;
    return out.str();
}

template<class T> inline std::string number(const std::optional<T>& value) {
    return value ? number(*value) : "NA";
}

// Modular subtraction also handles hardware timestamps with fewer than 64 bits.
inline double timestampMilliseconds(uint64_t begin, uint64_t end,
                                    uint32_t validBits, double periodNs) {
    if (!validBits || validBits > 64 || !(periodNs > 0.0))
        throw std::invalid_argument("Unsupported timestamp counter");
    uint64_t delta = end - begin;
    if (validBits < 64) delta &= (uint64_t{1} << validBits) - 1;
    return static_cast<double>(delta) * periodNs / 1e6;
}

struct Statistics {
    size_t valid = 0, missing = 0;
    std::optional<double> mean, p95;
};

inline Statistics statistics(std::vector<double> values, size_t total) {
    values.erase(std::remove_if(values.begin(), values.end(),
        [](double x) { return !std::isfinite(x) || x < 0; }), values.end());
    Statistics result{values.size(), total - values.size(), {}, {}};
    if (values.empty()) return result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    // Nearest-rank empirical percentile: sorted[ceil(0.95 * N) - 1].
    result.p95 = values[static_cast<size_t>(std::ceil(0.95 * values.size())) - 1];
    return result;
}

class Capture {
public:
    enum class State { Idle, Warmup, Sampling, Draining, Ready, Finished };

    void start(double warmupSeconds, size_t targetSamples, std::string scene,
               std::string pose, Metadata metadata, Clock::time_point now) {
        if (active()) throw std::logic_error("A capture is already active");
        if (!std::isfinite(warmupSeconds) || warmupSeconds < 0 || !targetSamples
            || targetSamples > 1000000)
            throw std::invalid_argument("Invalid capture settings");
        m_samples.clear();
        m_samples.reserve(targetSamples);
        m_openSample.reset();
        m_outstandingGpu = 0;
        m_target = targetSamples;
        m_warmup = warmupSeconds;
        m_started = now;
        m_samplingStarted.reset();
        m_scene = std::move(scene);
        m_pose = std::move(pose);
        m_metadata = std::move(metadata);
        m_reason.clear();
        m_status.clear();
        m_exportPath.clear();
        m_exportFailed = false;
        m_state = State::Warmup;
    }

    // Called once at every application frame start, before event processing.
    // The outgoing interval belongs to the frame whose GPU work we recorded.
    void onFrameStart(uint64_t frameId, Clock::time_point now) {
        if (m_openSample) {
            auto& sample = m_samples[*m_openSample];
            if (frameId == sample.frameId + 1) {
                const double ms = std::chrono::duration<double, std::milli>(now - sample.start).count();
                if (std::isfinite(ms) && ms > 0) sample.frameMs = ms;
            }
            m_openSample.reset();
        }
        updateReady();
        if (m_state == State::Warmup &&
            std::chrono::duration<double>(now - m_started).count() >= m_warmup) {
            m_state = State::Sampling;
            m_samplingStarted = now;
        }
        if (m_state != State::Sampling) return;
        Sample sample;
        sample.frameId = frameId;
        sample.start = now;
        sample.gpu.frameId = frameId;
        m_samples.push_back(std::move(sample));
        m_openSample = m_samples.size() - 1;
        ++m_outstandingGpu;
        if (m_samples.size() == m_target) stop("sample_count_reached");
    }

    void acceptGpuResult(GpuFrameResult result) {
        auto it = std::lower_bound(m_samples.begin(), m_samples.end(), result.frameId,
            [](const Sample& s, uint64_t id) { return s.frameId < id; });
        // Late results from warm-up, cancelled captures, or previous captures
        // cannot populate a different row. Application IDs never restart.
        if (it == m_samples.end() || it->frameId != result.frameId || it->gpuResolved) return;
        it->gpu = std::move(result);
        it->gpuResolved = true;
        --m_outstandingGpu;
        updateReady();
    }

    void markGpuUnavailable(uint64_t frameId, const std::string& reason) {
        GpuFrameResult result;
        result.frameId = frameId;
        result.status = reason;
        acceptGpuResult(std::move(result));
    }

    void stop(std::string reason = "stopped_by_user") {
        if (m_state != State::Warmup && m_state != State::Sampling) return;
        m_reason = std::move(reason);
        m_state = State::Draining;
        updateReady();
    }

    void interrupt(std::string reason) {
        if (!active()) return;
        m_reason = std::move(reason);
        if (m_state == State::Warmup || m_state == State::Sampling) m_state = State::Draining;
        updateReady();
    }

    void cancel() {
        m_samples.clear();
        m_openSample.reset();
        m_outstandingGpu = 0;
        m_state = State::Idle;
        m_status = "Capture cancelled; no files written";
        m_exportFailed = false;
    }

    // Call after harvesting results using the application's EXISTING shutdown
    // device-idle wait. No fabricated final start-to-start interval on exit.
    void finishOnShutdown() {
        if (!active()) return;
        stop("application_exit");
        if (m_openSample) m_reason = "application_exit_before_next_frame_start";
        m_openSample.reset();
        for (auto& sample : m_samples) {
            if (!sample.gpuResolved) {
                sample.gpu.status = "unavailable_at_shutdown";
                sample.gpuResolved = true;
            }
        }
        m_outstandingGpu = 0;
        updateReady();
    }

    State state() const { return m_state; }
    bool active() const {
        return m_state != State::Idle && m_state != State::Finished;
    }
    bool ready() const { return m_state == State::Ready; }
    size_t sampleCount() const { return m_samples.size(); }
    size_t targetCount() const { return m_target; }
    size_t pendingGpuCount() const { return m_outstandingGpu; }
    bool exportFailed() const { return m_exportFailed; }
    const std::vector<Sample>& samples() const { return m_samples; }
    const std::string& status() const { return m_status; }
    const std::filesystem::path& exportPath() const { return m_exportPath; }
    double warmupRemaining(Clock::time_point now) const {
        return std::max(0.0, m_warmup - std::chrono::duration<double>(now - m_started).count());
    }

    bool exportCsv(const std::filesystem::path& root) {
        if (m_state != State::Ready && !(m_state == State::Finished && m_exportFailed)) return false;
        // Even a failed export is attempted only once automatically. Retain
        // samples in memory so the user can explicitly retry a different path.
        m_state = State::Finished;
        m_exportFailed = true;
        try {
            std::filesystem::create_directories(root);
            auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm utc{};
#ifdef _WIN32
            gmtime_s(&utc, &t);
#else
            gmtime_r(&t, &utc);
#endif
            std::ostringstream stamp;
            stamp << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
            const std::string base = stamp.str() + "_" + slug(m_scene) + "_" + slug(m_pose);
            size_t suffix = 0;
            do { m_exportPath = root / (base + "_" + std::to_string(suffix++)); }
            while (!std::filesystem::create_directory(m_exportPath));
            // A completion marker is written last; failed/partial exports
            // never look like a complete benchmark directory.
            writeRaw(m_exportPath / "samples.csv");
            writeSummary(m_exportPath / "summary.csv");
            writeMetadata(m_exportPath / "metadata.csv");
            auto complete = open(m_exportPath / "capture_complete.txt");
            complete << "Capture files successfully written.\n";
            complete.close();
            m_exportFailed = false;
            m_status = "Exported " + std::to_string(m_samples.size()) + " frames to " + m_exportPath.string();
            return true;
        } catch (const std::exception& e) {
            m_status = std::string("Export failed (samples retained for retry): ") + e.what();
            return false;
        }
    }

private:
    void updateReady() {
        if (m_state == State::Draining && !m_openSample && !m_outstandingGpu)
            m_state = State::Ready;
    }
    static std::string slug(const std::string& text) {
        std::string result;
        for (unsigned char c : text) {
            if (result.size() >= 48) break;
            result += ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '-' || c == '_') ? char(c) : '_';
        }
        return result.empty() ? "capture" : result;
    }
    static std::ofstream open(const std::filesystem::path& path) {
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.imbue(std::locale::classic());
        out.open(path, std::ios::binary);
        out << std::setprecision(17);
        return out;
    }
    void writeRaw(const std::filesystem::path& path) const {
        auto out = open(path);
        out << "frame_index,frame_ms,grouping_gpu_ms,renderer_gpu_ms,allocation_requests,stored_fragments,"
               "retained_fragments,node_pool_overflow,node_pool_overflowed,per_pixel_retention_losses,"
               "gpu_status,application_frame_id,clear_gpu_ms,collect_gpu_ms,sort_gpu_ms,"
               "init_parents_gpu_ms,extraction_gpu_ms,composite_gpu_ms\n";
        for (size_t i = 0; i < m_samples.size(); ++i) {
            const auto& s = m_samples[i]; const auto& g = s.gpu;
            out << i << ',' << number(s.frameMs) << ',' << number(g.groupingMs) << ','
                << number(g.rendererMs) << ',' << number(g.allocationRequests) << ','
                << number(g.storedFragments) << ',' << number(g.retainedFragments) << ','
                << number(g.nodePoolOverflow) << ','
                << (g.nodePoolOverflow ? (*g.nodePoolOverflow ? "1" : "0") : "NA") << ','
                << number(g.perPixelRetentionLosses) << ',' << csvCell(g.status) << ','
                << s.frameId << ',' << number(g.clearMs) << ',' << number(g.collectMs) << ','
                << number(g.sortMs) << ',' << number(g.initParentsMs) << ','
                << number(g.extractionMs) << ',' << number(g.compositeMs) << '\n';
        }
        out.close();
    }
    template<class Getter> Statistics stats(Getter getter) const {
        std::vector<double> values;
        values.reserve(m_samples.size());
        for (const auto& sample : m_samples) if (auto value = getter(sample)) values.push_back(*value);
        return statistics(std::move(values), m_samples.size());
    }
    void writeSummary(const std::filesystem::path& path) const {
        auto out = open(path);
        out << "metric,valid_samples,missing_samples,mean,p95,value,unit\n";
        auto timing = [&](const char* name, auto getter) {
            const auto st = stats(getter);
            out << name << ',' << st.valid << ',' << st.missing << ','
                << number(st.mean) << ',' << number(st.p95) << ",NA,ms\n";
        };
        timing("frame_ms", [](const Sample& s) { return s.frameMs; });
        timing("grouping_gpu_ms", [](const Sample& s) { return s.gpu.groupingMs; });
        timing("renderer_gpu_ms", [](const Sample& s) { return s.gpu.rendererMs; });
        timing("clear_gpu_ms", [](const Sample& s) { return s.gpu.clearMs; });
        timing("collect_gpu_ms", [](const Sample& s) { return s.gpu.collectMs; });
        timing("sort_gpu_ms", [](const Sample& s) { return s.gpu.sortMs; });
        timing("init_parents_gpu_ms", [](const Sample& s) { return s.gpu.initParentsMs; });
        timing("extraction_gpu_ms", [](const Sample& s) { return s.gpu.extractionMs; });
        timing("composite_gpu_ms", [](const Sample& s) { return s.gpu.compositeMs; });
        size_t valid = 0, slow = 0;
        double sum = 0;
        for (const auto& s : m_samples) if (s.frameMs) {
            ++valid; sum += *s.frameMs; if (*s.frameMs > 16.67) ++slow;
        }
        // The requested aggregate uses sample_count, so do not silently use
        // a smaller denominator/numerator if an interval is unavailable.
        const bool complete = valid == m_samples.size() && valid != 0 && sum > 0;
        const std::optional<double> fps = complete ? std::optional<double>(1000.0 * valid / sum) : std::nullopt;
        const std::optional<double> slowPercent = complete ? std::optional<double>(100.0 * slow / valid) : std::nullopt;
        out << "application_fps," << valid << ',' << m_samples.size() - valid
            << ",NA,NA," << number(fps) << ",frames/s\n";
        out << "frame_intervals_over_16_67_ms_percent," << valid << ',' << m_samples.size() - valid
            << ",NA,NA," << number(slowPercent) << ",percent\n";
        out.close();
    }
    void writeMetadata(const std::filesystem::path& path) const {
        auto out = open(path);
        out << "key,value\n";
        auto row = [&](const std::string& key, const std::string& value) {
            out << csvCell(key) << ',' << csvCell(value) << '\n';
        };
        row("schema_version", "1"); row("scene_label", m_scene); row("camera_pose_label", m_pose);
        row("warmup_seconds_requested", number(m_warmup));
        row("warmup_seconds_actual", m_samplingStarted ? number(std::chrono::duration<double>(*m_samplingStarted - m_started).count()) : "NA");
        row("sample_count_requested", number(m_target)); row("sample_count_actual", number(m_samples.size()));
        row("completion_reason", m_reason); row("missing_value", "NA");
        row("percentile_method", "nearest rank: sorted[ceil(0.95*N)-1], using available values");
        row("application_fps_formula", "1000*sample_count/sum(frame_ms); NA if any frame_ms is missing");
        row("frame_budget_ms", "16.67"); row("slow_interval_comparison", "strictly greater than 16.67 ms");
        row("frame_index", "zero-based capture row; application_frame_id identifies the GPU source frame");
        row("frame_ms_boundary", "steady_clock start(frame N+1)-start(frame N), before glfwPollEvents");
        row("frame_ms_includes", "event processing, CPU work, ImGui, normal fence/acquire/present waits, frame pacing");
        for (const auto& entry : m_metadata) row(entry.first, entry.second);
        out.close();
    }

    State m_state = State::Idle;
    std::vector<Sample> m_samples;
    std::optional<size_t> m_openSample;
    size_t m_target = 1000, m_outstandingGpu = 0;
    double m_warmup = 30.0;
    Clock::time_point m_started{};
    std::optional<Clock::time_point> m_samplingStarted;
    Metadata m_metadata;
    std::string m_scene, m_pose, m_reason, m_status;
    std::filesystem::path m_exportPath;
    bool m_exportFailed = false;
};
} // namespace performance
