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
        lutBuffers[bufIdx].interpolationMode = LayeredTransferFunction::InterpolationMode::CatmullRom;
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

    static int debugCounter = 0;
    const bool shouldDebug = (++debugCounter % 10000 == 0);

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
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    if (lut->extrapolationMode == LayeredTransferFunction::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index + 1, 0, TABLE_SIZE - 1);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];

        return y0 + t * (y1 - y0);
    }

    double y0, y1;
    if (index < 0) {
        const double slope = lut->data[1] - lut->data[0];
        y0 = lut->data[0] + slope * index;
    } else if (index >= TABLE_SIZE) {
        const double slope = lut->data[TABLE_SIZE - 1] - lut->data[TABLE_SIZE - 2];
        y0 = lut->data[TABLE_SIZE - 1] + slope * (index - TABLE_SIZE + 1);
    } else {
        y0 = lut->data[index];
    }

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

        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
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

    const double a0 = y3 - y2 - y0 + y1;
    const double a1 = y0 - y1 - a0;
    const double a2 = y2 - y0;
    const double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double AudioEngine::evaluateCatmullRom(const LUTBuffer* lut, double x) const {
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

double AudioEngine::interpolate4Samples(LayeredTransferFunction::InterpolationMode mode,
                                       double y0, double y1, double y2, double y3, double t) const {
    switch (mode) {
    case LayeredTransferFunction::InterpolationMode::Linear:
        return y1 + t * (y2 - y1);

    case LayeredTransferFunction::InterpolationMode::Cubic: {
        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    case LayeredTransferFunction::InterpolationMode::CatmullRom:
        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    default:
        return y1 + t * (y2 - y1);
    }
}

double AudioEngine::evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT,
                                     double x, double gainOld, double gainNew) const {
    // Mix samples BEFORE interpolation (saves one polynomial eval per sample)
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    const auto interpMode = newLUT->interpolationMode;
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

        return interpolate4Samples(interpMode, mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
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

    return interpolate4Samples(interpMode, mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
}

// LUTRendererThread Implementation

LUTRendererThread::LUTRendererThread(AudioEngine& audioEngine_,
                                     std::atomic<int>& workerTargetIdx,
                                     std::atomic<bool>& readyFlag,
                                     std::function<void()> visualizerCallback)
    : juce::Thread("LUTRendererThread")
    , audioEngine(audioEngine_)
    , workerTargetIndex(workerTargetIdx)
    , newLUTReady(readyFlag)
    , onVisualizerUpdate(std::move(visualizerCallback)) {
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
        // Always render both visualizer LUT (2K samples) and DSP LUT (16K samples)
        // Previously skipped DSP render during crossfade to save CPU, but this caused
        // bug where job was consumed without rendering (visualizer updates but audio
        // still plays old value). Fix: Always render. Audio thread defers pickup via
        // newLUTReady flag until crossfade completes.
        renderVisualizerLUT(latestJob);

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
        HarmonicLayer& harmonicLayer,
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

void LUTRendererThread::renderVisualizerLUT(const RenderJob& job) {
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

    std::array<double, VISUALIZER_LUT_SIZE> visualizerData;
    for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        visualizerData[i] = tempLTF->evaluateForRendering(x, normScalar);
    }

    if (visualizerLUTPtr && onVisualizerUpdate) {
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

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        outputBuffer->data[i] = tempLTF->evaluateForRendering(x, normScalar);
    }

    outputBuffer->version = job.version;
    outputBuffer->interpolationMode = job.interpolationMode;
    outputBuffer->extrapolationMode = job.extrapolationMode;

    newLUTReady.store(true, std::memory_order_release);
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

// TransferFunctionDirtyPoller Implementation

TransferFunctionDirtyPoller::TransferFunctionDirtyPoller(LayeredTransferFunction& ltf_,
                                                         LUTRendererThread& renderer_)
    : ltf(ltf_)
    , renderer(renderer_) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
}

TransferFunctionDirtyPoller::~TransferFunctionDirtyPoller() {
    stopTimer();
}

void TransferFunctionDirtyPoller::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const uint64_t currentVersion = ltf.getVersion();

    if (currentVersion != lastSeenVersion) {
        const RenderJob job = captureRenderJob();
        renderer.enqueueJob(job);
        lastSeenVersion = currentVersion;
    }
}

void TransferFunctionDirtyPoller::forceRender() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const RenderJob job = captureRenderJob();
    renderer.enqueueJob(job);
    lastSeenVersion = ltf.getVersion();
}

RenderJob TransferFunctionDirtyPoller::captureRenderJob() {
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

    job.interpolationMode = ltf.getInterpolationMode();
    job.extrapolationMode = ltf.getExtrapolationMode();
    job.version = ltf.getVersion();

    return job;
}

} // namespace dsp_core
