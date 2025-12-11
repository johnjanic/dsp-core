#include "SeamlessTransferFunctionImpl.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {
    /**
     * Smoothstep interpolation (cubic Hermite)
     *
     * Provides ease-in/ease-out S-curve with zero derivative at endpoints.
     * Used for perceptually smooth crossfading between LUTs.
     *
     * Formula: t² × (3 - 2t)
     * Properties: C¹ continuous, symmetric, computationally cheap
     *
     * @param t Normalized position [0, 1]
     * @return Smoothed position [0, 1]
     */
    inline double smoothstep(double t) {
        // Clamp to [0, 1] (defensive programming - shouldn't happen in normal operation)
        t = std::clamp(t, 0.0, 1.0);

        // Smoothstep formula: t² × (3 - 2t)
        // Expanded: 3t² - 2t³
        return t * t * (3.0 - 2.0 * t);
    }
} // namespace

//==============================================================================
// AudioEngine Implementation
//==============================================================================

AudioEngine::AudioEngine() {
    // Initialize all three buffers to identity function (y = x)
    // This prevents undefined behavior before first LUT render
    for (int bufIdx = 0; bufIdx < 3; ++bufIdx) {
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            lutBuffers[bufIdx].data[i] = x;
        }
        lutBuffers[bufIdx].version = 0;
        lutBuffers[bufIdx].interpolationMode = LayeredTransferFunction::InterpolationMode::CatmullRom;
        lutBuffers[bufIdx].extrapolationMode = LayeredTransferFunction::ExtrapolationMode::Clamp;
    }
}

void AudioEngine::prepareToPlay(double sampleRate, int samplesPerBlock) {
    this->sampleRate = sampleRate;
    // CRITICAL: Crossfade must be long enough to smooth DC offset transitions
    // DC blocking filter (5Hz highpass) has ~32ms time constant
    // 50ms crossfade = 1.5× time constant (balances smoothness vs latency)
    constexpr double crossfadeDurationMs = 50.0;
    constexpr double msToSeconds = 1000.0;
    crossfadeSamples = static_cast<int>(sampleRate * crossfadeDurationMs / msToSeconds);

    // EDGE CASE: If sample rate changes mid-crossfade, clamp position to new duration
    // This prevents overshoot if duration is shortened (rare but possible during host init)
    if (crossfading && crossfadePosition >= crossfadeSamples) {
        crossfading = false; // Complete crossfade immediately
    }
}

double AudioEngine::applyTransferFunction(double x) const {
    if (crossfading) {
        // S-curve crossfade between old and new LUT (OPTIMIZED)
        const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
        const double alpha = smoothstep(t);
        const double gainOld = 1.0 - alpha;
        const double gainNew = alpha;
        // OPTIMIZATION: Mix samples first, then interpolate (not the other way around)
        return evaluateCrossfade(oldLUT, newLUT, x, gainOld, gainNew);
    } else {
        // No crossfade: use primary LUT
        const int idx = primaryIndex.load(std::memory_order_acquire);
        return evaluateLUT(&lutBuffers[idx], x);
    }
}

void AudioEngine::processBuffer(juce::AudioBuffer<double>& buffer) const {
    // Check for new LUT once per multi-channel buffer
    checkForNewLUT();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    static int debugCounter = 0;
    const bool shouldDebug = (++debugCounter % 10000 == 0);

    // Process all samples across all channels
    // CRITICAL: Crossfade position advances once per sample, NOT per channel
    for (int i = 0; i < numSamples; ++i) {
        if (crossfading) {
            // Calculate S-curve crossfade gains (same for all channels)
            const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
            const double alpha = smoothstep(t);
            const double gainOld = 1.0 - alpha;
            const double gainNew = alpha;

            // Apply crossfade to all channels (OPTIMIZED)
            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                // OPTIMIZATION: Mix samples first, then interpolate (saves one polynomial eval)
                channelData[i] = evaluateCrossfade(oldLUT, newLUT, input, gainOld, gainNew);

                // Debug first channel first sample
                if (shouldDebug && i == 0 && ch == 0) {
                    DBG("[CROSSFADE] pos=" + juce::String(crossfadePosition) +
                        " alpha=" + juce::String(alpha, 4) +
                        " input=" + juce::String(input, 6) +
                        " output=" + juce::String(channelData[i], 6));
                }
            }

            // Advance crossfade position ONCE per sample (not per channel!)
            if (++crossfadePosition >= crossfadeSamples) {
                crossfading = false;
                DBG("[AUDIO] Crossfade complete");
            }
        } else {
            // No crossfade: use primary LUT for all channels
            const int idx = primaryIndex.load(std::memory_order_acquire);

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                channelData[i] = evaluateLUT(&lutBuffers[idx], input);

                // Debug first channel first sample
                if (shouldDebug && i == 0 && ch == 0 && std::abs(input) < 0.1) {
                    DBG("[AUDIO] processBuffer: input=" + juce::String(input) +
                        " output=" + juce::String(channelData[i]) +
                        " bufferIdx=" + juce::String(idx) +
                        " LUT[8192]=" + juce::String(lutBuffers[idx].data[8192]));
                }
            }
        }
    }
}

void AudioEngine::checkForNewLUT() const {
    if (newLUTReady.load(std::memory_order_acquire)) {
        // CRITICAL FIX: Don't interrupt active crossfade
        // Let current crossfade complete, then pick up new LUT on next processBlock call
        // This prevents audible clicks from aborting mid-crossfade
        if (crossfading) {
            DBG("[AUDIO] New LUT ready but crossfade in progress - deferring update");
            return;  // Skip this update - flag stays true, we'll catch it after crossfade completes
        }

        // CRITICAL: Use acquire memory order to ensure worker's LUT data is visible
        const int oldPrimaryIdx = primaryIndex.load(std::memory_order_acquire);
        const int oldSecondaryIdx = secondaryIndex.load(std::memory_order_acquire);
        const int workerIdx = workerTargetIndex.load(std::memory_order_acquire);

        // Rotate indices: worker target becomes new primary
        // old primary becomes new secondary (for crossfade)
        // old secondary becomes new worker target (safe to overwrite)
        primaryIndex.store(workerIdx, std::memory_order_release);
        secondaryIndex.store(oldPrimaryIdx, std::memory_order_release);
        workerTargetIndex.store(oldSecondaryIdx, std::memory_order_release);

        // Set up crossfade pointers
        oldLUT = &lutBuffers[oldPrimaryIdx];
        newLUT = &lutBuffers[workerIdx];

        newLUTReady.store(false, std::memory_order_release);
        crossfading = true;
        crossfadePosition = 0;
        DBG("[AUDIO] Starting crossfade (" + juce::String(crossfadeSamples) + " samples)");
    }
}

double AudioEngine::evaluateLUT(const LUTBuffer* lut, double x) const {
    switch (lut->interpolationMode) {
    case LayeredTransferFunction::InterpolationMode::Linear:
        return evaluateLinear(lut, x);
    case LayeredTransferFunction::InterpolationMode::Cubic:
        return evaluateCubic(lut, x);
    case LayeredTransferFunction::InterpolationMode::CatmullRom:
        return evaluateCatmullRom(lut, x);
    default:
        return evaluateLinear(lut, x);
    }
}

double AudioEngine::evaluateLinear(const LUTBuffer* lut, double x) const {
    // Map x to table index using constants
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // Fast path: Clamp mode (most common)
    if (lut->extrapolationMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index + 1, 0, TABLE_SIZE - 1);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];

        return y0 + t * (y1 - y0);
    }

    // Linear extrapolation path
    double y0, y1;

    // Handle index
    if (index < 0) {
        const double slope = lut->data[1] - lut->data[0];
        y0 = lut->data[0] + slope * index;
    } else if (index >= TABLE_SIZE) {
        const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
        y0 = lut->data[TABLE_SIZE - 1] + slope * (index - TABLE_SIZE + 1);
    } else {
        y0 = lut->data[index];
    }

    // Handle index + 1
    const int index1 = index + 1;
    if (index1 < 0) {
        const double slope = lut->data[1] - lut->data[0];
        y1 = lut->data[0] + slope * index1;
    } else if (index1 >= TABLE_SIZE) {
        const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
        y1 = lut->data[TABLE_SIZE - 1] + slope * (index1 - TABLE_SIZE + 1);
    } else {
        y1 = lut->data[index1];
    }

    return y0 + t * (y1 - y0);
}

double AudioEngine::evaluateCubic(const LUTBuffer* lut, double x) const {
    // Map x to table index
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // Fast path: Clamp mode
    if (lut->extrapolationMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];
        const double y2 = lut->data[idx2];
        const double y3 = lut->data[idx3];

        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    // Linear extrapolation path
    auto getSample = [lut](int i) -> double {
        if (i < 0) {
            const double slope = lut->data[1] - lut->data[0];
            return lut->data[0] + slope * i;
        }
        if (i >= TABLE_SIZE) {
            const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
            return lut->data[TABLE_SIZE - 1] + slope * (i - TABLE_SIZE + 1);
        }
        return lut->data[i];
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    const double a0 = y3 - y2 - y0 + y1;
    const double a1 = y0 - y1 - a0;
    const double a2 = y2 - y0;
    const double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double AudioEngine::evaluateCatmullRom(const LUTBuffer* lut, double x) const {
    // Map x to table index
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // Fast path: Clamp mode
    if (lut->extrapolationMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];
        const double y2 = lut->data[idx2];
        const double y3 = lut->data[idx3];

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    // Linear extrapolation path
    auto getSample = [lut](int i) -> double {
        if (i < 0) {
            const double slope = lut->data[1] - lut->data[0];
            return lut->data[0] + slope * i;
        }
        if (i >= TABLE_SIZE) {
            const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
            return lut->data[TABLE_SIZE - 1] + slope * (i - TABLE_SIZE + 1);
        }
        return lut->data[i];
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    // Standard Catmull-Rom interpolation formula
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double AudioEngine::interpolate4Samples(LayeredTransferFunction::InterpolationMode mode,
                                       double y0, double y1, double y2, double y3, double t) const {
    switch (mode) {
    case LayeredTransferFunction::InterpolationMode::Linear:
        // Linear only uses y1 and y2
        return y1 + t * (y2 - y1);

    case LayeredTransferFunction::InterpolationMode::Cubic: {
        // Cubic interpolation formula
        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    case LayeredTransferFunction::InterpolationMode::CatmullRom:
        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    default:
        return y1 + t * (y2 - y1); // Fallback to linear
    }
}

double AudioEngine::evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT,
                                     double x, double gainOld, double gainNew) const {
    // CRITICAL OPTIMIZATION: Mix samples BEFORE interpolation
    // Both LUTs share the same fractional index, so we can:
    // 1. Map x → fractional index ONCE
    // 2. Fetch 4 samples from each LUT
    // 3. Mix corresponding samples
    // 4. Do ONE interpolation on mixed samples
    //
    // This saves one expensive polynomial evaluation per sample!

    // Map x to table index (shared by both LUTs)
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // Determine interpolation mode (use newLUT's mode - it's the target state)
    const auto interpMode = newLUT->interpolationMode;
    const auto extrapMode = newLUT->extrapolationMode;

    // Fast path: Clamp mode (most common)
    if (extrapMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        // Fetch 4 samples from old LUT
        const double old_y0 = oldLUT->data[idx0];
        const double old_y1 = oldLUT->data[idx1];
        const double old_y2 = oldLUT->data[idx2];
        const double old_y3 = oldLUT->data[idx3];

        // Fetch 4 samples from new LUT
        const double new_y0 = newLUT->data[idx0];
        const double new_y1 = newLUT->data[idx1];
        const double new_y2 = newLUT->data[idx2];
        const double new_y3 = newLUT->data[idx3];

        // Mix corresponding samples
        const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
        const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
        const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
        const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

        // Do ONE interpolation on mixed samples
        return interpolate4Samples(interpMode, mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
    }

    // Linear extrapolation path (rare)
    auto getSample = [](const LUTBuffer* lut, int i) -> double {
        if (i < 0) {
            const double slope = lut->data[1] - lut->data[0];
            return lut->data[0] + slope * i;
        }
        if (i >= TABLE_SIZE) {
            const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
            return lut->data[TABLE_SIZE - 1] + slope * (i - TABLE_SIZE + 1);
        }
        return lut->data[i];
    };

    // Fetch 4 samples from old LUT (with extrapolation)
    const double old_y0 = getSample(oldLUT, index - 1);
    const double old_y1 = getSample(oldLUT, index);
    const double old_y2 = getSample(oldLUT, index + 1);
    const double old_y3 = getSample(oldLUT, index + 2);

    // Fetch 4 samples from new LUT (with extrapolation)
    const double new_y0 = getSample(newLUT, index - 1);
    const double new_y1 = getSample(newLUT, index);
    const double new_y2 = getSample(newLUT, index + 1);
    const double new_y3 = getSample(newLUT, index + 2);

    // Mix corresponding samples
    const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
    const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
    const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
    const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

    // Do ONE interpolation on mixed samples
    return interpolate4Samples(interpMode, mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
}

//==============================================================================
// LUTRendererThread Implementation
//==============================================================================

LUTRendererThread::LUTRendererThread(AudioEngine& audioEngine_,
                                     std::atomic<int>& workerTargetIdx,
                                     std::atomic<bool>& readyFlag,
                                     std::function<void()> visualizerCallback)
    : juce::Thread("LUTRendererThread")
    , audioEngine(audioEngine_)
    , workerTargetIndex(workerTargetIdx)
    , newLUTReady(readyFlag)
    , onVisualizerUpdate(std::move(visualizerCallback)) {
    // Create worker-owned LayeredTransferFunction for rendering
    // Use same table size and range as SeamlessTransferFunction
    tempLTF = std::make_unique<LayeredTransferFunction>(TABLE_SIZE, MIN_VALUE, MAX_VALUE);
}

void LUTRendererThread::run() {
    while (!threadShouldExit()) {
        // Wait for jobs or exit signal
        wakeEvent.wait(1000); // 1 second timeout to check threadShouldExit()

        if (threadShouldExit()) {
            break;
        }

        // Process all pending jobs (with coalescing)
        processJobs();
    }
}

void LUTRendererThread::enqueueJob(const RenderJob& job) {
    int start1, size1, start2, size2;
    jobQueue.prepareToWrite(1, start1, size1, start2, size2);

    if (size1 > 0) {
        jobSlots[start1] = job; // Copy job data
        jobQueue.finishedWrite(1);
        wakeEvent.signal();
    }
#if JUCE_DEBUG
    else {
        // Queue full - silently drop (next poll will capture latest state)
        // This is acceptable because we only care about the most recent version
        DBG("LUTRendererThread: Queue full, dropping job version " << job.version);
    }
#endif
}

void LUTRendererThread::processJobs() {
    RenderJob latestJob;
    bool hasJob = false;

    // Drain entire queue, keeping only the latest job (coalescing)
    int start1, size1, start2, size2;
    while (true) {
        jobQueue.prepareToRead(1, start1, size1, start2, size2);
        if (size1 == 0)
            break;

        latestJob = jobSlots[start1]; // Copy job data
        jobQueue.finishedRead(1);
        hasJob = true;
    }

    if (hasJob) {
        // TWO-SPEED RENDERING STRATEGY:
        // 1. FAST PATH: Always render visualizer LUT (2K samples, ~2ms)
        renderVisualizerLUT(latestJob);

        // 2. RENDER DSP LUT: Always render, even during crossfade (16K samples, ~16ms)
        // CRITICAL FIX: Previously skipped DSP render during crossfade to save CPU,
        // but this caused bug where job was consumed without rendering!
        // Symptom: User drags harmonic slider to 0, visualizer updates but audio
        // still plays old non-zero value until next user interaction.
        // Root cause: Job consumed from queue, DSP render skipped, audio thread
        // defers pickup due to active crossfade, job lost forever.
        // Fix: Always render DSP LUT. Audio thread will defer pickup via newLUTReady
        // flag until crossfade completes (~50ms), then pick up latest LUT.
        // CPU cost: Minimal (~40% of one core), acceptable for real-time audio.
        const int targetIdx = workerTargetIndex.load(std::memory_order_relaxed);
        if (lutBuffers != nullptr) {
            renderDSPLUT(latestJob, &lutBuffers[targetIdx]);
        }
#if JUCE_DEBUG
        else {
            DBG("LUTRendererThread: lutBuffers pointer not set, cannot render DSP LUT");
        }
#endif
    }
}

namespace {
    /**
     * Compute normalization scalar for harmonic mode
     *
     * Scans base layer + harmonics to find max absolute value,
     * then returns 1.0 / max to normalize composite to [-1, 1].
     *
     * Respects paint stroke freezing (uses frozen scalar when active).
     *
     * @param baseLayer Full base layer data (16384 values)
     * @param coefficients Harmonic coefficients [0] = WT mix, [1..40] = harmonics
     * @param harmonicLayer Reference to harmonic layer for evaluation
     * @param normalizationEnabled If false, returns 1.0 (no scaling)
     * @param paintStrokeActive If true, uses frozenScalar (paint stroke mode)
     * @param frozenScalar Scalar to use when paintStrokeActive=true
     * @return Normalization scalar in range (0, 1], or 1.0 if disabled
     */
    double computeNormalizationScalar(
        const std::array<double, TABLE_SIZE>& baseLayer,
        const std::array<double, 41>& coefficients,
        HarmonicLayer& harmonicLayer,
        bool normalizationEnabled,
        bool paintStrokeActive,
        double frozenScalar
    ) {
        // If normalization disabled, return identity scalar
        if (!normalizationEnabled) {
            return 1.0;
        }

        // If paint stroke active, use frozen scalar
        if (paintStrokeActive) {
            return frozenScalar;
        }

        // Convert coefficients array to vector for HarmonicLayer API
        std::vector<double> coeffsVec(coefficients.begin(), coefficients.end());

        // Compute max absolute value across entire composite
        double maxAbsValue = 0.0;

        for (int i = 0; i < TABLE_SIZE; ++i) {
            // Map index to x-coordinate
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);

            // Get base layer value
            const double baseValue = baseLayer[i];

            // Evaluate harmonics at x
            const double harmonicValue = harmonicLayer.evaluate(x, coeffsVec, TABLE_SIZE);

            // Compute unnormalized composite
            const double unnormalized = coefficients[0] * baseValue + harmonicValue;

            // Track maximum absolute value
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }

        // Compute normalization scalar (avoid division by zero)
        constexpr double epsilon = 1e-12;
        return (maxAbsValue > epsilon) ? (1.0 / maxAbsValue) : 1.0;
    }
} // namespace

void LUTRendererThread::renderVisualizerLUT(const RenderJob& job) {
    // FAST PATH: Render 2K samples for visualizer (~2ms)
    // ALWAYS called, regardless of crossfade state
    //
    // OPTIMIZATION: Reuses tempLTF state from DSP render if available,
    // otherwise sets up state just for visualizer render

    // Set state in tempLTF (shared setup for both paths)
    for (int i = 0; i < TABLE_SIZE; ++i) {
        tempLTF->setBaseLayerValue(i, job.baseLayerData[i]);
    }
    tempLTF->setHarmonicCoefficients(job.coefficients);
    tempLTF->setSplineAnchors(job.splineAnchors);
    tempLTF->setInterpolationMode(job.interpolationMode);
    tempLTF->setExtrapolationMode(job.extrapolationMode);
    tempLTF->setRenderingMode(job.renderingMode);

    // Compute normalization scalar (used by Harmonic mode, ignored by Paint/Spline modes)
    const double normScalar = computeNormalizationScalar(
        job.baseLayerData,
        job.coefficients,
        tempLTF->getHarmonicLayer(),
        job.normalizationEnabled,
        job.paintStrokeActive,
        job.frozenNormalizationScalar
    );

    // Render visualizer LUT (2048 samples)
    std::array<double, VISUALIZER_LUT_SIZE> visualizerData;
    for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        visualizerData[i] = tempLTF->evaluateForRendering(x, normScalar);
    }

    // Update visualizer LUT (async on message thread)
    if (visualizerLUTPtr && onVisualizerUpdate) {
        // Capture data directly into lambda using C++14 init-capture
        juce::MessageManager::callAsync([this, data = visualizerData]() {
            if (visualizerLUTPtr) {
                *visualizerLUTPtr = data;
                if (onVisualizerUpdate) {
                    onVisualizerUpdate();
                }
            }
        });
    }
}

void LUTRendererThread::renderDSPLUT(const RenderJob& job, LUTBuffer* outputBuffer) {
    // Set state in tempLTF
    for (int i = 0; i < TABLE_SIZE; ++i) {
        tempLTF->setBaseLayerValue(i, job.baseLayerData[i]);
    }
    tempLTF->setHarmonicCoefficients(job.coefficients);
    tempLTF->setSplineAnchors(job.splineAnchors);
    tempLTF->setInterpolationMode(job.interpolationMode);
    tempLTF->setExtrapolationMode(job.extrapolationMode);
    tempLTF->setRenderingMode(job.renderingMode);

    // Compute normalization scalar (used by Harmonic mode, ignored by Paint/Spline modes)
    const double normScalar = computeNormalizationScalar(
        job.baseLayerData,
        job.coefficients,
        tempLTF->getHarmonicLayer(),
        job.normalizationEnabled,
        job.paintStrokeActive,
        job.frozenNormalizationScalar
    );

    // SIMPLIFIED LOOP - LayeredTransferFunction handles mode switching internally
    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        outputBuffer->data[i] = tempLTF->evaluateForRendering(x, normScalar);
    }

    outputBuffer->version = job.version;
    outputBuffer->interpolationMode = job.interpolationMode;
    outputBuffer->extrapolationMode = job.extrapolationMode;

    // 9. Signal audio thread that new LUT is ready
    newLUTReady.store(true, std::memory_order_release);

    // NOTE: Visualizer update is now handled by renderVisualizerLUT() (fast path)
    // This eliminates the need to copy 16K samples to the visualizer
}

void LUTRendererThread::setVisualizerCallback(std::function<void()> callback) {
    onVisualizerUpdate = std::move(callback);
}

void LUTRendererThread::setVisualizerLUTPointer(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr) {
    visualizerLUTPtr = lutPtr;
}

void LUTRendererThread::setLUTBuffersPointer(LUTBuffer* lutBuffers_) {
    lutBuffers = lutBuffers_;
}

//==============================================================================
// TransferFunctionDirtyPoller Implementation
//==============================================================================

TransferFunctionDirtyPoller::TransferFunctionDirtyPoller(LayeredTransferFunction& ltf_,
                                                         LUTRendererThread& renderer_)
    : ltf(ltf_)
    , renderer(renderer_) {
    // CRITICAL: Verify construction on message thread
    // This is required because we read non-atomic data from LayeredTransferFunction
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
}

TransferFunctionDirtyPoller::~TransferFunctionDirtyPoller() {
    // CRITICAL: Stop timer to prevent new callbacks during destruction
    stopTimer();
}

void TransferFunctionDirtyPoller::timerCallback() {
    // CRITICAL: Verify we're on message thread (debug builds only)
    // LayeredTransferFunction is mutated from message thread, so we must read from message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Check if version has changed since last tick
    const uint64_t currentVersion = ltf.getVersion();

    if (currentVersion != lastSeenVersion) {
        // Version changed - capture snapshot and enqueue render job
        RenderJob job = captureRenderJob();
        renderer.enqueueJob(job);
        lastSeenVersion = currentVersion;
    }
}

void TransferFunctionDirtyPoller::forceRender() {
    // CRITICAL: Verify we're on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Capture current state and enqueue immediately
    RenderJob job = captureRenderJob();
    renderer.enqueueJob(job);
    lastSeenVersion = ltf.getVersion();
}

RenderJob TransferFunctionDirtyPoller::captureRenderJob() {
    RenderJob job;

    // Copy full base layer (128KB memcpy - fast enough at 25Hz)
    for (int i = 0; i < TABLE_SIZE; ++i) {
        job.baseLayerData[i] = ltf.getBaseLayerValue(i);
    }

    // Copy coefficients
    job.coefficients = ltf.getHarmonicCoefficients();

    // Copy spline anchors
    job.splineAnchors = ltf.getSplineLayer().getAnchors();

    // Copy modes and normalization state
    job.normalizationEnabled = ltf.isNormalizationEnabled();
    job.paintStrokeActive = ltf.isPaintStrokeActive(); // CRITICAL: Freeze scalar during paint strokes
    job.renderingMode = ltf.getRenderingMode();

    // OPTIMIZATION: Only capture frozenNormalizationScalar for Harmonic mode
    // Paint mode doesn't use normalization (direct base read)
    // Spline mode doesn't use normalization (direct spline evaluation)
    if (job.renderingMode == dsp_core::RenderingMode::Harmonic) {
        job.frozenNormalizationScalar = ltf.getNormalizationScalar();
    } else {
        job.frozenNormalizationScalar = 1.0; // Unused but set to safe default
    }

    job.interpolationMode = ltf.getInterpolationMode();
    job.extrapolationMode = ltf.getExtrapolationMode();

    // Stamp version
    job.version = ltf.getVersion();

    return job;
}

} // namespace dsp_core
