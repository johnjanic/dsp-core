#pragma once

#include "LayeredTransferFunction.h"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core {

// Forward declarations
struct RenderJob;
class LUTRendererThread;
class TransferFunctionDirtyPoller;

//==============================================================================
// Constants (hardcoded for performance)
//==============================================================================

static constexpr int TABLE_SIZE = 16384;          // DSP LUT size (audio thread)
static constexpr int VISUALIZER_LUT_SIZE = 2048;  // Visualizer LUT size (UI thread)
static constexpr double MIN_VALUE = -1.0;
static constexpr double MAX_VALUE = 1.0;

//==============================================================================
// LUTBuffer - Triple-buffered lookup table
//==============================================================================

struct LUTBuffer {
    std::array<double, TABLE_SIZE> data;
    uint64_t version{0};
    LayeredTransferFunction::InterpolationMode interpolationMode{
        LayeredTransferFunction::InterpolationMode::Linear};
    LayeredTransferFunction::ExtrapolationMode extrapolationMode{
        LayeredTransferFunction::ExtrapolationMode::Clamp};
};

//==============================================================================
// AudioEngine - Triple-buffered audio thread component with crossfade
//==============================================================================

/**
 * AudioEngine - Audio thread component for seamless transfer function updates
 *
 * Architecture:
 *   - Triple buffering prevents data race during crossfade
 *   - lutBuffers[0,1]: Used for crossfading (audio thread reads)
 *   - lutBuffers[2]: Worker thread writes here (safe from audio thread)
 *   - 10ms linear crossfade (sample-rate adaptive)
 *
 * Thread Safety:
 *   - Audio thread: reads primaryIndex, secondaryIndex, checks newLUTReady
 *   - Worker thread: writes to lutBuffers[workerTargetIndex], sets newLUTReady
 *   - No locks, all communication via atomics
 *
 * Memory Ordering:
 *   - acquire/release for index swaps (ensures LUT data visibility)
 *   - relaxed for worker target (worker thread only)
 */
class AudioEngine {
  public:
    AudioEngine();

    /**
     * Prepare for playback (audio thread)
     *
     * Calculates sample-rate-adaptive crossfade duration.
     * Called from audio thread in prepareToPlay().
     *
     * @param sampleRate Sample rate in Hz
     * @param samplesPerBlock Maximum block size
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock);

    /**
     * Apply transfer function to single sample (audio thread)
     *
     * Uses active LUT or crossfades between old and new LUT.
     *
     * @param x Input sample
     * @return Output sample
     */
    double applyTransferFunction(double x) const;

    /**
     * Process multi-channel buffer in-place (audio thread)
     *
     * Processes all channels with shared crossfade state.
     * Checks for new LUT once at start, then processes all channels.
     * Crossfade position advances correctly (once per sample, not per channel).
     *
     * @param buffer Multi-channel audio buffer (modified in-place)
     */
    void processBuffer(juce::AudioBuffer<double>& buffer) const;

    /**
     * Get reference to worker target index (for worker thread)
     *
     * Worker thread loads this index to know where to write.
     *
     * @return Atomic reference to worker target index
     */
    std::atomic<int>& getWorkerTargetIndexReference() {
        return workerTargetIndex;
    }

    /**
     * Get reference to new LUT ready flag (for worker thread)
     *
     * Worker thread sets this to true after rendering LUT.
     *
     * @return Atomic reference to ready flag
     */
    std::atomic<bool>& getNewLUTReadyFlag() {
        return newLUTReady;
    }

    /**
     * Get LUT buffers pointer (for worker thread writes)
     *
     * Worker thread writes to lutBuffers[workerTargetIndex].
     *
     * @return Pointer to LUT buffer array
     */
    LUTBuffer* getLUTBuffers() {
        return lutBuffers;
    }

    /**
     * Check if audio engine is currently crossfading (for worker thread)
     *
     * Worker thread uses this to decide whether to render DSP LUT:
     * - If crossfading: Skip DSP render (audio thread can't accept new LUT)
     * - If not crossfading: Render DSP LUT (audio thread ready)
     *
     * This implements the two-speed worker strategy:
     * - Always render visualizer LUT (2K samples, ~2ms)
     * - Only render DSP LUT when audio thread can accept it
     *
     * @return true if crossfade in progress, false otherwise
     */
    bool isCrossfading() const {
        return crossfading;
    }

  private:
    /**
     * Check for new LUT from worker thread (audio thread)
     *
     * Called once per processBlock(). If new LUT is ready:
     *   1. Aborts any active crossfade
     *   2. Rotates buffer indices (worker → primary, primary → secondary, secondary → worker)
     *   3. Starts new crossfade from old primary to new primary
     *
     * Uses acquire memory ordering to ensure worker's LUT data is visible.
     */
    void checkForNewLUT() const;

    /**
     * Evaluate LUT with interpolation/extrapolation
     *
     * @param lut LUT buffer to evaluate
     * @param x Input value
     * @return Interpolated output value
     */
    double evaluateLUT(const LUTBuffer* lut, double x) const;

    /**
     * Linear interpolation helper (fast path for most common case)
     */
    double evaluateLinear(const LUTBuffer* lut, double x) const;

    /**
     * Cubic interpolation helper
     */
    double evaluateCubic(const LUTBuffer* lut, double x) const;

    /**
     * Catmull-Rom interpolation helper
     */
    double evaluateCatmullRom(const LUTBuffer* lut, double x) const;

    // TRIPLE BUFFERING (prevents data race during crossfade):
    // - lutBuffers[0,1]: Used for crossfading (audio thread reads)
    // - lutBuffers[2]: Worker thread writes here (safe from audio thread)
    LUTBuffer lutBuffers[3];

    // Atomics are mutable because they're modified in const methods (thread-safe state)
    mutable std::atomic<int> primaryIndex{0};      // Active LUT for playback
    mutable std::atomic<int> secondaryIndex{1};    // Previous LUT (used during crossfade)
    mutable std::atomic<int> workerTargetIndex{2}; // Worker writes here
    mutable std::atomic<bool> newLUTReady{false};

    // Crossfade state (audio thread local - mutable for const methods)
    double sampleRate{44100.0};
    mutable int crossfadeSamples{441};  // Recalculated in prepareToPlay()
    mutable int crossfadePosition{0};
    mutable std::atomic<bool> crossfading{false};  // Atomic for worker thread reads
    mutable const LUTBuffer* oldLUT{nullptr};
    mutable const LUTBuffer* newLUT{nullptr};
};

//==============================================================================
// RenderJob - Self-contained snapshot of LayeredTransferFunction state
//==============================================================================

/**
 * RenderJob - Self-contained transfer function state for worker thread rendering
 *
 * Design Principle: NO ContentStore dependency
 *   - Full base layer data copied (128KB memcpy - acceptable at 25Hz)
 *   - Spline anchors copied (not referenced)
 *   - Preserves dsp-core module purity
 *
 * Paint Stroke Normalization Handling:
 *   - paintStrokeActive = true: Use frozenNormalizationScalar directly
 *   - paintStrokeActive = false: Recompute scalar during rendering
 *   - This distinction is CRITICAL for correct paint stroke rendering
 *
 * Interpolation/Extrapolation Modes:
 *   - Version-tracked even though they only affect lookup (not LUT contents)
 *   - Changing these modes triggers full LUT re-render for simplicity
 *   - Acceptable because: modes rarely change, 25Hz coalescing minimizes thrashing
 *   - Intentional simplicity trade-off vs distinguishing "shape" vs "lookup" changes
 */
struct RenderJob {
    // Full base layer data (self-contained, no hash/lookup needed)
    std::array<double, TABLE_SIZE> baseLayerData;

    // Coefficients (WT mix + 40 harmonics = 41 total)
    std::array<double, 41> coefficients;

    // Spline anchors (copied, not referenced)
    std::vector<SplineAnchor> splineAnchors;

    // Mode flags
    bool normalizationEnabled{false};
    bool paintStrokeActive{false}; // CRITICAL: Distinguishes frozen vs computed scalar

    // Frozen normalization scalar (used when paintStrokeActive = true)
    double frozenNormalizationScalar{1.0};

    // Interpolation/extrapolation modes
    LayeredTransferFunction::InterpolationMode interpolationMode{
        LayeredTransferFunction::InterpolationMode::Linear};
    LayeredTransferFunction::ExtrapolationMode extrapolationMode{
        LayeredTransferFunction::ExtrapolationMode::Clamp};

    // Rendering mode (determines evaluation path: Paint/Harmonic/Spline)
    RenderingMode renderingMode{RenderingMode::Paint};

    // Version stamp
    uint64_t version{0};
};

//==============================================================================
// LUTRendererThread - Worker thread for asynchronous LUT rendering
//==============================================================================

/**
 * LUTRendererThread - Background worker thread that renders LUTs from RenderJobs
 *
 * Architecture:
 *   - Lock-free job queue (AbstractFifo, 4 slots - sufficient for 25Hz polling)
 *   - Job coalescing: drains queue, renders only latest job
 *   - Worker-owned LayeredTransferFunction for isolated rendering
 *   - Writes to lutBuffers[workerTargetIndex] (safe from audio thread)
 *   - Updates visualizer via MessageManager::callAsync
 *
 * Queue Overflow:
 *   - Silently drops jobs if queue full (acceptable - next poll will retry)
 *   - Debug logging for overflow events
 *   - We only care about the most recent version
 *
 * Thread Safety:
 *   - Worker thread: reads from job queue, writes to workerTargetIndex buffer
 *   - Audio thread: never touches workerTargetIndex buffer (safe isolation)
 *   - Message thread: receives visualizer updates via callAsync
 *
 * Self-Contained RenderJobs:
 *   - No ContentStore dependency (preserves dsp-core module boundary)
 *   - Full state snapshot (base layer, coefficients, spline anchors)
 *   - Renders into temporary LayeredTransferFunction
 */
class LUTRendererThread : public juce::Thread {
  public:
    /**
     * Construct worker thread
     *
     * @param audioEngine Reference to AudioEngine (to check crossfade state)
     * @param workerTargetIdx Reference to atomic index (where to write LUT)
     * @param readyFlag Reference to atomic flag (signals audio thread)
     * @param visualizerCallback Optional callback for visualizer repaints
     */
    LUTRendererThread(AudioEngine& audioEngine,
                      std::atomic<int>& workerTargetIdx,
                      std::atomic<bool>& readyFlag,
                      std::function<void()> visualizerCallback = nullptr);

    /**
     * Worker thread main loop
     *
     * Waits for jobs, processes with coalescing, renders LUTs.
     * Runs until stopThread() is called.
     */
    void run() override;

    /**
     * Enqueue render job from message thread
     *
     * Lock-free enqueue with overflow protection.
     * If queue is full, silently drops job (next poll will capture latest state).
     *
     * @param job RenderJob to enqueue (copied into queue)
     */
    void enqueueJob(const RenderJob& job);

    /**
     * Set visualizer callback (message thread only)
     *
     * Called after LUT render completes (via MessageManager::callAsync).
     *
     * @param callback Callback function for visualizer repaint
     */
    void setVisualizerCallback(std::function<void()> callback);

    /**
     * Set visualizer LUT buffer pointer (message thread only)
     *
     * Worker thread will write to this buffer via MessageManager::callAsync.
     *
     * @param lutPtr Pointer to UI-owned visualizer LUT buffer (VISUALIZER_LUT_SIZE samples)
     */
    void setVisualizerLUTPointer(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr);

    /**
     * Get pointer to LUT buffers (for direct worker writes)
     *
     * Worker thread writes to lutBuffers[workerTargetIndex].
     *
     * @return Pointer to LUT buffer array (from AudioEngine)
     */
    void setLUTBuffersPointer(LUTBuffer* lutBuffers);

  private:
    /**
     * Process all pending jobs with coalescing
     *
     * Drains entire queue, keeping only the latest job.
     * Renders that job into workerTargetIndex buffer.
     *
     * TWO-SPEED RENDERING STRATEGY:
     *   1. FAST PATH: Always render visualizer LUT (2K samples, ~2ms)
     *      - Updates UI immediately via MessageManager::callAsync
     *   2. SLOW PATH: Conditionally render DSP LUT (16K samples, ~16ms)
     *      - Only renders if !audioEngine.isCrossfading()
     *      - Skips render if audio thread is busy crossfading
     *   3. CPU savings: Skip 16K render when DSP can't accept new LUT
     */
    void processJobs();

    /**
     * Render visualizer LUT from self-contained RenderJob (FAST PATH)
     *
     * Renders 2K samples for UI display (~2ms).
     * ALWAYS called, regardless of crossfade state.
     * Updates visualizer buffer via MessageManager::callAsync.
     *
     * @param job RenderJob containing full state snapshot
     */
    void renderVisualizerLUT(const RenderJob& job);

    /**
     * Render DSP LUT from self-contained RenderJob (SLOW PATH)
     *
     * Renders 16K samples for audio processing (~16ms).
     * ONLY called if !audioEngine.isCrossfading().
     * Writes to lutBuffers[workerTargetIndex].
     *
     * Workflow:
     *   1. Restore base layer from job.baseLayerData
     *   2. Set coefficients from job.coefficients
     *   3. Set spline anchors from job.splineAnchors
     *   4. Set interpolation/extrapolation modes
     *   5. Set rendering mode from job.renderingMode (Paint/Harmonic/Spline)
     *   6. Compute normalization scalar (Harmonic mode only)
     *   7. Render LUT via evaluateForRendering()
     *   8. Copy LUT to output buffer
     *   9. Signal audio thread (newLUTReady = true)
     *
     * @param job RenderJob containing full state snapshot
     * @param outputBuffer LUT buffer to write into (lutBuffers[workerTargetIndex])
     */
    void renderDSPLUT(const RenderJob& job, LUTBuffer* outputBuffer);

    // Job queue (lock-free, 4 slots is sufficient given 25Hz polling + job coalescing)
    juce::WaitableEvent wakeEvent;
    juce::AbstractFifo jobQueue{4};
    RenderJob jobSlots[4];

    // Worker-owned LayeredTransferFunction for rendering
    std::unique_ptr<LayeredTransferFunction> tempLTF;

    // Reference to AudioEngine (for checking crossfade state)
    AudioEngine& audioEngine;

    // Atomic references (shared with AudioEngine)
    std::atomic<int>& workerTargetIndex;
    std::atomic<bool>& newLUTReady;

    // LUT buffers pointer (from AudioEngine)
    LUTBuffer* lutBuffers{nullptr};

    // Visualizer integration
    std::function<void()> onVisualizerUpdate;
    std::array<double, VISUALIZER_LUT_SIZE>* visualizerLUTPtr{nullptr};
};

//==============================================================================
// TransferFunctionDirtyPoller - 25Hz version change detector
//==============================================================================

/**
 * TransferFunctionDirtyPoller - Timer-based version change detector
 *
 * THREADING CONTRACT (CRITICAL):
 *   - MUST run on message thread (JUCE Timer contract)
 *   - LayeredTransferFunction is mutated from message thread (via controller)
 *   - Reads non-atomic data (coefficients vector, base layer)
 *   - Constructor asserts isThisTheMessageThread()
 *   - timerCallback() asserts message thread in debug builds
 *   - DO NOT override timer thread behavior
 *
 * Architecture:
 *   - Polls LayeredTransferFunction.getVersion() at 25Hz (40ms interval)
 *   - Detects version changes (version != lastSeenVersion)
 *   - Captures self-contained RenderJob snapshot
 *   - Enqueues job to LUTRendererThread
 *   - NO ContentStore dependency (preserves module boundaries)
 *
 * Performance:
 *   - <100μs overhead per tick
 *   - Full snapshot capture: ~128KB memcpy (acceptable at 25Hz)
 *   - CPU overhead: <0.1%
 *
 * Initial Render:
 *   - forceRender() triggers immediate render (used for initialization)
 *   - Ensures visualizer shows correct curve on startup
 */
class TransferFunctionDirtyPoller : public juce::Timer {
  public:
    /**
     * Construct dirty poller (message thread only)
     *
     * CRITICAL: Asserts that construction happens on message thread.
     * This is required because we read non-atomic data from LayeredTransferFunction.
     *
     * @param ltf LayeredTransferFunction to poll (editing model)
     * @param renderer LUTRendererThread to enqueue jobs to
     */
    TransferFunctionDirtyPoller(LayeredTransferFunction& ltf, LUTRendererThread& renderer);

    /**
     * Destructor - cancels any pending async callbacks
     *
     * CRITICAL: Sets destructed flag to prevent use-after-free from pending callbacks.
     */
    ~TransferFunctionDirtyPoller();

    /**
     * Timer callback - polls version and enqueues jobs (message thread)
     *
     * Called at 25Hz (40ms interval).
     * Checks if version has changed since last tick.
     * If changed, captures RenderJob and enqueues to worker thread.
     *
     * CRITICAL: Asserts message thread in debug builds.
     */
    void timerCallback() override;

    /**
     * Force immediate render (message thread only)
     *
     * Used for initial setup - ensures first LUT render happens immediately.
     * Captures current state and enqueues job without waiting for next tick.
     * Updates lastSeenVersion to prevent duplicate render on next tick.
     */
    void forceRender();

  private:
    /**
     * Capture self-contained RenderJob snapshot
     *
     * Creates complete snapshot of LayeredTransferFunction state:
     *   - Full base layer (128KB memcpy)
     *   - Coefficients array
     *   - Spline anchors vector (copy)
     *   - Mode flags (spline, normalization, deferred)
     *   - Frozen normalization scalar
     *   - Interpolation/extrapolation modes
     *   - Version stamp
     *
     * NO ContentStore dependency - fully self-contained.
     *
     * @return RenderJob containing full state snapshot
     */
    RenderJob captureRenderJob();

    LayeredTransferFunction& ltf;
    LUTRendererThread& renderer;
    uint64_t lastSeenVersion{0};
};

} // namespace dsp_core
