#pragma once

#include "LayeredTransferFunction.h"
#include <array>
#include <functional>
#include <memory>

namespace dsp_core {

/**
 * SeamlessTransferFunction - Reusable click-free transfer function with async LUT rendering
 *
 * GOAL: Eliminate clicks/pops during ALL transfer function edits while maintaining
 *       <50ms latency for interactive operations.
 *
 * ARCHITECTURE: Three-thread design with 25Hz polling
 *   - Message thread: UI/controller mutates editing model, 25Hz poller detects changes
 *   - Worker thread: Renders LUTs asynchronously from self-contained snapshots
 *   - Audio thread: Triple-buffered LUT with 10ms linear crossfade (sample-rate-adaptive)
 *
 * KEY DESIGN DECISIONS:
 *   - Self-contained RenderJobs with NO ContentStore dependency (preserves dsp-core module purity)
 *   - Triple buffering prevents data race during crossfade
 *   - Job coalescing (only renders latest version)
 *   - Hardcoded constants (16384 table size, -1.0 to 1.0 range) for performance
 *   - Visualizer shows latest rendered LUT (target curve audio is crossfading toward)
 *
 * PERFORMANCE:
 *   - Latency: <50ms for interactive operations (harmonic sliders, paint strokes)
 *   - CPU overhead: <0.5% (25Hz polling + worker thread)
 *   - Memory overhead: ~1.4MB (393KB triple-buffered LUTs + ~1MB job queue)
 *
 * REUSABILITY:
 *   - Drop-in replacement for LayeredTransferFunction
 *   - Can be reused across multiple Aspen Instruments plugins
 *   - No plugin-specific dependencies
 */
class SeamlessTransferFunction {
  public:
    // Constants (hardcoded for performance - not configurable)
    static constexpr int TABLE_SIZE = 16384;
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;

    /**
     * INITIALIZATION SEQUENCE (REQUIRED ORDER):
     *
     * 1. Construct SeamlessTransferFunction
     *    - editingModel initialized to identity
     *    - AudioEngine initialized to identity LUTs
     *    - Worker/poller NOT started yet
     *
     * 2. Create controller: new TransferFunctionController(stf.getEditingModel())
     *    - Controller can now mutate editing model
     *    - Still no async updates happening
     *
     * 3. Call startSeamlessUpdates() (message thread)
     *    - Starts worker thread and 25Hz poller
     *    - Triggers initial LUT render for visualizer
     *    - From this point, edits trigger async LUT renders
     *    - MUST be called AFTER controller creation
     *
     * 4. Call prepareToPlay(sampleRate, samplesPerBlock) (audio thread)
     *    - Sets up crossfade duration based on sample rate
     *    - Can be called before or after startSeamlessUpdates()
     *
     * 5. Normal operation: process audio, make edits
     *
     * 6. Call releaseResources() before destruction
     *    - Stops worker thread cleanly
     *
     * EXAMPLE (from PluginProcessor):
     *
     *   // Constructor (message thread):
     *   transferFunction = std::make_unique<SeamlessTransferFunction>();
     *   controller = std::make_unique<TransferFunctionController>(
     *       transferFunction->getEditingModel()
     *   );
     *   transferFunction->startSeamlessUpdates();  // NOW safe to start
     *
     *   // prepareToPlay (audio thread):
     *   transferFunction->prepareToPlay(sampleRate, samplesPerBlock);
     *
     *   // releaseResources (audio thread):
     *   transferFunction->releaseResources();
     */
    SeamlessTransferFunction();
    ~SeamlessTransferFunction();

    // Non-copyable, non-movable (manages threads)
    SeamlessTransferFunction(const SeamlessTransferFunction&) = delete;
    SeamlessTransferFunction& operator=(const SeamlessTransferFunction&) = delete;
    SeamlessTransferFunction(SeamlessTransferFunction&&) = delete;
    SeamlessTransferFunction& operator=(SeamlessTransferFunction&&) = delete;

    /**
     * Access editing model (for UI/controller, message thread only)
     *
     * The editing model is what the controller mutates in response to user input.
     * Changes are detected by the poller and asynchronously rendered to LUTs.
     *
     * @return Reference to editing model
     */
    LayeredTransferFunction& getEditingModel();
    const LayeredTransferFunction& getEditingModel() const;

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
     * Process block of samples in-place (audio thread)
     *
     * Checks for new LUT once at start, then processes all samples.
     * Crossfade state advances sample-by-sample if active.
     *
     * @param samples Pointer to sample buffer (modified in-place)
     * @param numSamples Number of samples to process
     */
    void processBlock(double* samples, int numSamples) const;

    /**
     * Prepare for playback (audio thread)
     *
     * Calculates sample-rate-adaptive crossfade duration.
     * Can be called before or after startSeamlessUpdates().
     *
     * @param sampleRate Sample rate in Hz
     * @param samplesPerBlock Maximum block size
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock);

    /**
     * Release resources (audio thread)
     *
     * Delegates to stopSeamlessUpdates() asynchronously on message thread.
     * Safe to call from audio thread.
     */
    void releaseResources();

    /**
     * Start seamless updates (message thread, after controller created)
     *
     * Starts worker thread and 25Hz poller.
     * Triggers initial LUT render for visualizer.
     * MUST be called AFTER controller creation.
     *
     * Asserts:
     *   - Called on message thread
     *   - Not already started
     */
    void startSeamlessUpdates();

    /**
     * Stop seamless updates (message thread)
     *
     * Stops poller and worker thread.
     * Safe to call multiple times (idempotent).
     */
    void stopSeamlessUpdates();

    /**
     * Notify that editing model has changed (message thread)
     *
     * Call this when the editing model's version has incremented to trigger
     * an immediate LUT render. This is called by the UI timer (25Hz) when
     * it detects version changes.
     *
     * Thread-safe: Can be called from message thread.
     */
    void notifyEditingModelChanged();

    /**
     * Get visualizer LUT (message thread only)
     *
     * Returns the latest rendered LUT (target curve that audio is crossfading toward
     * or has reached). This may be ahead of what's currently playing if a crossfade
     * is still in progress.
     *
     * @return Const reference to visualizer LUT buffer
     */
    const std::array<double, TABLE_SIZE>& getVisualizerLUT() const;

    /**
     * Set visualizer callback (message thread only)
     *
     * Called after LUT render completes (via MessageManager::callAsync).
     * Use this to trigger visualizer repaint.
     *
     * @param callback Callback function for visualizer repaint
     */
    void setVisualizerCallback(std::function<void()> callback);

  private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace dsp_core
