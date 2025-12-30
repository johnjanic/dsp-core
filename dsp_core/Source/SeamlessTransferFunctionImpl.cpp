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

// AudioEngine Implementation

AudioEngine::AudioEngine() {
    for (int bufIdx = 0; bufIdx < 3; ++bufIdx) {
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            lutBuffers[bufIdx].data[i] = x;
        }
        lutBuffers[bufIdx].version = 0;
        lutBuffers[bufIdx].extrapolationMode = LayeredTransferFunction::ExtrapolationMode::Clamp;
    }
}

void AudioEngine::prepareToPlay(double sampleRate, int samplesPerBlock) {
    this->sampleRate = sampleRate;
    // 50ms crossfade = 1.5× DC blocking time constant (balances smoothness vs latency)
    constexpr double crossfadeDurationMs = 50.0;
    constexpr double msToSeconds = 1000.0;
    crossfadeSamples = static_cast<int>(sampleRate * crossfadeDurationMs / msToSeconds);

    if (crossfading && crossfadePosition >= crossfadeSamples) {
        crossfading = false;
    }
}

double AudioEngine::applyTransferFunction(double x) const {
    if (crossfading) {
        const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
        const double alpha = smoothstep(t);
        const double gainOld = 1.0 - alpha;
        const double gainNew = alpha;
        return evaluateCrossfade(oldLUT, newLUT, x, gainOld, gainNew);
    } else {
        const int idx = primaryIndex.load(std::memory_order_acquire);
        return evaluateLUT(&lutBuffers[idx], x);
    }
}

void AudioEngine::processBuffer(juce::AudioBuffer<double>& buffer) const {
    checkForNewLUT();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Crossfade position advances once per sample (not per channel)
    for (int i = 0; i < numSamples; ++i) {
        if (crossfading) {
            const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
            const double alpha = smoothstep(t);
            const double gainOld = 1.0 - alpha;
            const double gainNew = alpha;

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                channelData[i] = evaluateCrossfade(oldLUT, newLUT, input, gainOld, gainNew);
            }

            if (++crossfadePosition >= crossfadeSamples) {
                crossfading = false;
            }
        } else {
            const int idx = primaryIndex.load(std::memory_order_acquire);

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                channelData[i] = evaluateLUT(&lutBuffers[idx], input);
            }
        }
    }
}

void AudioEngine::checkForNewLUT() const {
    if (newLUTReady.load(std::memory_order_acquire)) {
        // Don't interrupt active crossfade (prevents audible clicks)
        if (crossfading) {
            return;
        }

        const int oldPrimaryIdx = primaryIndex.load(std::memory_order_acquire);
        const int oldSecondaryIdx = secondaryIndex.load(std::memory_order_acquire);
        const int workerIdx = workerTargetIndex.load(std::memory_order_acquire);

        primaryIndex.store(workerIdx, std::memory_order_release);
        secondaryIndex.store(oldPrimaryIdx, std::memory_order_release);
        workerTargetIndex.store(oldSecondaryIdx, std::memory_order_release);

        oldLUT = &lutBuffers[oldPrimaryIdx];
        newLUT = &lutBuffers[workerIdx];

        newLUTReady.store(false, std::memory_order_release);
        crossfading = true;
        crossfadePosition = 0;
    }
}

double AudioEngine::evaluateLUT(const LUTBuffer* lut, double x) const {
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

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

double AudioEngine::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) {
    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double AudioEngine::evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT,
                                     double x, double gainOld, double gainNew) const {
    // Mix samples BEFORE interpolation (saves one polynomial eval per sample)
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    const auto extrapMode = newLUT->extrapolationMode;

    if (extrapMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        const double old_y0 = oldLUT->data[idx0];
        const double old_y1 = oldLUT->data[idx1];
        const double old_y2 = oldLUT->data[idx2];
        const double old_y3 = oldLUT->data[idx3];

        const double new_y0 = newLUT->data[idx0];
        const double new_y1 = newLUT->data[idx1];
        const double new_y2 = newLUT->data[idx2];
        const double new_y3 = newLUT->data[idx3];

        const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
        const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
        const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
        const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

        return interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
    }

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

    const double old_y0 = getSample(oldLUT, index - 1);
    const double old_y1 = getSample(oldLUT, index);
    const double old_y2 = getSample(oldLUT, index + 1);
    const double old_y3 = getSample(oldLUT, index + 2);

    const double new_y0 = getSample(newLUT, index - 1);
    const double new_y1 = getSample(newLUT, index);
    const double new_y2 = getSample(newLUT, index + 1);
    const double new_y3 = getSample(newLUT, index + 2);

    const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
    const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
    const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
    const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

    return interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
}

// LUTRendererThread Implementation

LUTRendererThread::LUTRendererThread(AudioEngine& audioEngine_,
                                     std::atomic<int>& workerTargetIdx,
                                     std::atomic<bool>& readyFlag)
    : juce::Thread("LUTRendererThread")
    , audioEngine(audioEngine_)
    , workerTargetIndex(workerTargetIdx)
    , newLUTReady(readyFlag) {
    tempLTF = std::make_unique<LayeredTransferFunction>(TABLE_SIZE, MIN_VALUE, MAX_VALUE);
}

void LUTRendererThread::run() {
    while (!threadShouldExit()) {
        wakeEvent.wait(1000);

        if (threadShouldExit()) {
            break;
        }

        processJobs();
    }
}

void LUTRendererThread::enqueueJob(const RenderJob& job) {
    int start1, size1, start2, size2;
    jobQueue.prepareToWrite(1, start1, size1, start2, size2);

    if (size1 > 0) {
        jobSlots[start1] = job;
        jobQueue.finishedWrite(1);
        wakeEvent.signal();
    }
#if JUCE_DEBUG
    else {
        // Queue full - drop job (next poll will capture latest state)
    }
#endif
}

void LUTRendererThread::processJobs() {
    RenderJob latestJob;
    bool hasJob = false;

    int start1, size1, start2, size2;
    while (true) {
        jobQueue.prepareToRead(1, start1, size1, start2, size2);
        if (size1 == 0) {
            break;
        }

        latestJob = jobSlots[start1];
        jobQueue.finishedRead(1);
        hasJob = true;
    }

    if (hasJob) {
        // Render DSP LUT only (16K samples)
        // Visualizer is handled separately by VisualizerUpdateTimer (60Hz direct model reads)
        const int targetIdx = workerTargetIndex.load(std::memory_order_relaxed);
        if (lutBuffers != nullptr) {
            renderDSPLUT(latestJob, &lutBuffers[targetIdx]);
        }
    }
}

namespace {
    double computeNormalizationScalar(
        const std::array<double, TABLE_SIZE>& baseLayer,
        const std::array<double, 41>& coefficients,
        const HarmonicLayer& harmonicLayer, // Changed to const reference
        bool normalizationEnabled,
        bool paintStrokeActive,
        double frozenScalar
    ) {
        if (!normalizationEnabled) {
            return 1.0;
        }

        if (paintStrokeActive) {
            return frozenScalar;
        }

        const std::vector<double> coeffsVec(coefficients.begin(), coefficients.end());
        double maxAbsValue = 0.0;

        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            const double baseValue = baseLayer[i];
            const double harmonicValue = harmonicLayer.evaluate(x, coeffsVec, TABLE_SIZE);
            const double unnormalized = coefficients[0] * baseValue + harmonicValue;
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }

        constexpr double epsilon = 1e-12;
        return (maxAbsValue > epsilon) ? (1.0 / maxAbsValue) : 1.0;
    }
} // namespace

void LUTRendererThread::renderDSPLUT(const RenderJob& job, LUTBuffer* outputBuffer) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        tempLTF->setBaseLayerValue(i, job.baseLayerData[i]);
    }
    tempLTF->setHarmonicCoefficients(job.coefficients);
    tempLTF->setSplineAnchors(job.splineAnchors);
    tempLTF->setExtrapolationMode(job.extrapolationMode);
    tempLTF->setRenderingMode(job.renderingMode);

    // Compute normalization scalar (used by Harmonic mode, ignored by Paint/Spline modes)
    const double normScalar = computeNormalizationScalar(
        job.baseLayerData,
        job.coefficients,
        static_cast<const HarmonicLayer&>(tempLTF->getHarmonicLayer()), // Cast to const
        job.normalizationEnabled,
        job.paintStrokeActive,
        job.frozenNormalizationScalar
    );

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        outputBuffer->data[i] = tempLTF->evaluateForRendering(x, normScalar);
    }

    outputBuffer->version = job.version;
    outputBuffer->extrapolationMode = job.extrapolationMode;

    newLUTReady.store(true, std::memory_order_release);
}

void LUTRendererThread::setLUTBuffersPointer(LUTBuffer* buffers) {
    lutBuffers = buffers;
}

// LUTRenderTimer Implementation (20Hz, DSP LUT only, guaranteed delivery)

LUTRenderTimer::LUTRenderTimer(LayeredTransferFunction& ltf_,
                               LUTRendererThread& renderer_)
    : ltf(ltf_)
    , renderer(renderer_) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startTimerHz(20);  // 50ms interval, matches crossfade timing
}

LUTRenderTimer::~LUTRenderTimer() {
    stopTimer();
}

void LUTRenderTimer::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const uint64_t currentVersion = ltf.getVersion();

    // Track version changes
    if (currentVersion != lastSeenVersion) {
        lastSeenVersion = currentVersion;
    }

    // Render if we're behind - handles BOTH intermediate AND final renders
    // This two-version tracking guarantees the final change is never skipped
    if (lastRenderedVersion != lastSeenVersion) {
        const RenderJob job = captureRenderJob();
        renderer.enqueueJob(job);
        lastRenderedVersion = lastSeenVersion;
    }
}

void LUTRenderTimer::forceRender() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const RenderJob job = captureRenderJob();
    renderer.enqueueJob(job);

    // Update both versions to prevent duplicate render on next tick
    const uint64_t currentVersion = ltf.getVersion();
    lastSeenVersion = currentVersion;
    lastRenderedVersion = currentVersion;
}

RenderJob LUTRenderTimer::captureRenderJob() {
    RenderJob job;

    for (int i = 0; i < TABLE_SIZE; ++i) {
        job.baseLayerData[i] = ltf.getBaseLayerValue(i);
    }

    job.coefficients = ltf.getHarmonicCoefficients();
    job.splineAnchors = ltf.getSplineLayer().getAnchors();
    job.normalizationEnabled = ltf.isNormalizationEnabled();
    job.paintStrokeActive = ltf.isPaintStrokeActive();
    job.renderingMode = ltf.getRenderingMode();

    if (job.renderingMode == dsp_core::RenderingMode::Harmonic) {
        job.frozenNormalizationScalar = ltf.getNormalizationScalar();
    } else {
        job.frozenNormalizationScalar = 1.0;
    }

    job.extrapolationMode = ltf.getExtrapolationMode();
    job.version = ltf.getVersion();

    return job;
}

// VisualizerUpdateTimer Implementation (120Hz, direct model reads)

VisualizerUpdateTimer::VisualizerUpdateTimer(LayeredTransferFunction& model)
    : editingModel(model) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startTimerHz(120);  // 120Hz for smooth UI updates during drag
}

VisualizerUpdateTimer::~VisualizerUpdateTimer() {
    stopTimer();
}

void VisualizerUpdateTimer::setVisualizerTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr,
                                                 std::function<void()> callback) {
    visualizerLUTPtr = lutPtr;
    onVisualizerUpdate = std::move(callback);
}

void VisualizerUpdateTimer::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const uint64_t currentVersion = editingModel.getVersion();

    if (currentVersion != lastSeenVersion) {
        lastSeenVersion = currentVersion;

        if (visualizerLUTPtr) {
            // Update normalization scalar for Harmonic mode (must be computed fresh when harmonics change)
            // Paint mode: uses frozen scalar during strokes (controller manages this)
            // Spline mode: doesn't use normalization scalar
            if (editingModel.getRenderingMode() == RenderingMode::Harmonic &&
                editingModel.isNormalizationEnabled() &&
                !editingModel.isPaintStrokeActive()) {
                editingModel.updateNormalizationScalar();
            }

            // Get normalization scalar for rendering
            const double normScalar = editingModel.getNormalizationScalar();

            // Direct model sampling - ~0.5ms for 1024 points
            for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
                const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1))
                                           * (MAX_VALUE - MIN_VALUE);
                (*visualizerLUTPtr)[i] = editingModel.evaluateForRendering(x, normScalar);
            }

            if (onVisualizerUpdate) {
                onVisualizerUpdate();
            }
        }
    }
}

void VisualizerUpdateTimer::forceUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (visualizerLUTPtr) {
        // Update normalization scalar for Harmonic mode (must be computed fresh when harmonics change)
        if (editingModel.getRenderingMode() == RenderingMode::Harmonic &&
            editingModel.isNormalizationEnabled() &&
            !editingModel.isPaintStrokeActive()) {
            editingModel.updateNormalizationScalar();
        }

        // Get normalization scalar for rendering
        const double normScalar = editingModel.getNormalizationScalar();

        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1))
                                       * (MAX_VALUE - MIN_VALUE);
            (*visualizerLUTPtr)[i] = editingModel.evaluateForRendering(x, normScalar);
        }

        if (onVisualizerUpdate) {
            onVisualizerUpdate();
        }
    }

    lastSeenVersion = editingModel.getVersion();
}

} // namespace dsp_core
