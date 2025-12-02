#include "SeamlessTransferFunctionImpl.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

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
        lutBuffers[bufIdx].interpolationMode = LayeredTransferFunction::InterpolationMode::Linear;
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
        // Linear crossfade between old and new LUT
        const double alpha = static_cast<double>(crossfadePosition) / crossfadeSamples;
        const double gainOld = 1.0 - alpha;
        const double gainNew = alpha;
        return gainOld * evaluateLUT(oldLUT, x) + gainNew * evaluateLUT(newLUT, x);
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
            // Calculate crossfade gains (same for all channels)
            const double alpha = static_cast<double>(crossfadePosition) / crossfadeSamples;
            const double gainOld = 1.0 - alpha;
            const double gainNew = alpha;

            // Apply crossfade to all channels
            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                const double oldValue = evaluateLUT(oldLUT, input);
                const double newValue = evaluateLUT(newLUT, input);
                channelData[i] = gainOld * oldValue + gainNew * newValue;

                // Debug first channel first sample
                if (shouldDebug && i == 0 && ch == 0) {
                    DBG("[CROSSFADE] pos=" + juce::String(crossfadePosition) +
                        " alpha=" + juce::String(alpha, 4) +
                        " input=" + juce::String(input, 6) +
                        " old=" + juce::String(oldValue, 6) +
                        " new=" + juce::String(newValue, 6) +
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

//==============================================================================
// LUTRendererThread Implementation
//==============================================================================

LUTRendererThread::LUTRendererThread(std::atomic<int>& workerTargetIdx,
                                     std::atomic<bool>& readyFlag,
                                     std::function<void()> visualizerCallback)
    : juce::Thread("LUTRendererThread")
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
        // Get worker target buffer index (safe - audio thread won't touch this buffer)
        const int targetIdx = workerTargetIndex.load(std::memory_order_relaxed);

        // Render LUT into target buffer
        if (lutBuffers != nullptr) {
            renderLUT(latestJob, &lutBuffers[targetIdx]);
        }
#if JUCE_DEBUG
        else {
            DBG("LUTRendererThread: lutBuffers pointer not set, cannot render LUT");
        }
#endif
    }
}

void LUTRendererThread::renderLUT(const RenderJob& job, LUTBuffer* outputBuffer) {
    // 1. Restore base layer from job data
    for (int i = 0; i < TABLE_SIZE; ++i) {
        tempLTF->setBaseLayerValue(i, job.baseLayerData[i]);
    }

    // 2. Set coefficients
    tempLTF->setHarmonicCoefficients(job.coefficients);

    // 3. Set spline anchors (use wrapper method for consistency)
    tempLTF->setSplineAnchors(job.splineAnchors);

    // 4. Set spline layer mode
    tempLTF->setSplineLayerEnabled(job.splineLayerEnabled);

    // 5. CRITICAL: Handle deferred normalization state
    if (job.normalizationEnabled) {
        tempLTF->setNormalizationEnabled(true);
        if (job.deferNormalization) {
            // Restore frozen normalization scalar
            tempLTF->setNormalizationScalar(job.frozenNormalizationScalar);
            tempLTF->setDeferNormalization(true);
        } else {
            // Normal normalization (will compute scalar)
            tempLTF->setDeferNormalization(false);
        }
    } else {
        // Normalization disabled (scalar locked to 1.0)
        tempLTF->setNormalizationEnabled(false);
    }

    // 6. Set interpolation/extrapolation modes
    tempLTF->setInterpolationMode(job.interpolationMode);
    tempLTF->setExtrapolationMode(job.extrapolationMode);

    // 7. Render LUT based on mode
    if (job.splineLayerEnabled) {
        // SPLINE MODE: Direct evaluation (bypass compositeTable)
        // Spline evaluation is direct (no normalization layer), so read directly from SplineLayer
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            outputBuffer->data[i] = tempLTF->getSplineLayer().evaluate(x);
        }
    } else {
        // HARMONIC MODE: Use compositeTable (normalization included)
        // Harmonic mode needs normalization, which happens in updateComposite()
        tempLTF->updateComposite();
        for (int i = 0; i < TABLE_SIZE; ++i) {
            outputBuffer->data[i] = tempLTF->getCompositeValue(i);
        }
    }

    outputBuffer->version = job.version;
    outputBuffer->interpolationMode = job.interpolationMode;
    outputBuffer->extrapolationMode = job.extrapolationMode;

    // 9. Signal audio thread that new LUT is ready
    newLUTReady.store(true, std::memory_order_release);

    // 10. Update visualizer LUT (async on message thread)
    if (visualizerLUTPtr && onVisualizerUpdate) {
        // Capture data directly into lambda using C++14 init-capture (eliminates one copy)
        // Copy is still needed because outputBuffer may be overwritten by next render
        juce::MessageManager::callAsync([this, data = outputBuffer->data]() {
            if (visualizerLUTPtr) {
                *visualizerLUTPtr = data;
                if (onVisualizerUpdate) {
                    onVisualizerUpdate();
                }
            }
        });
    }
}

void LUTRendererThread::setVisualizerCallback(std::function<void()> callback) {
    onVisualizerUpdate = std::move(callback);
}

void LUTRendererThread::setVisualizerLUTPointer(std::array<double, TABLE_SIZE>* lutPtr) {
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
    job.splineLayerEnabled = ltf.isSplineLayerEnabled();
    job.normalizationEnabled = ltf.isNormalizationEnabled();
    job.deferNormalization = ltf.isNormalizationDeferred(); // CRITICAL: Capture deferred state
    job.frozenNormalizationScalar = ltf.getNormalizationScalar();
    job.interpolationMode = ltf.getInterpolationMode();
    job.extrapolationMode = ltf.getExtrapolationMode();

    // Stamp version
    job.version = ltf.getVersion();

    return job;
}

} // namespace dsp_core
