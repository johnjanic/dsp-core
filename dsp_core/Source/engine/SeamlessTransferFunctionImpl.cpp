#include "SeamlessTransferFunctionImpl.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>

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
    constexpr double msToSeconds = 1000.0;
    crossfadeSamples = static_cast<int>(sampleRate * SeamlessConfig::CROSSFADE_DURATION_MS / msToSeconds);

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

void AudioEngine::processBuffer(platform::AudioBuffer<double>& buffer) const {
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

LUTRendererThread::LUTRendererThread(AudioEngine& audioEngine,
                                     std::atomic<int>& workerTargetIdx,
                                     std::atomic<bool>& readyFlag)
    : audioEngine_(audioEngine)
    , workerTargetIndex_(workerTargetIdx)
    , newLUTReady_(readyFlag) {
    tempLTF_ = std::make_unique<LayeredTransferFunction>(TABLE_SIZE, MIN_VALUE, MAX_VALUE);
}

LUTRendererThread::~LUTRendererThread() {
    stopThread(2000);
}

void LUTRendererThread::startThread() {
    shouldExit_.store(false, std::memory_order_release);
    thread_ = std::thread([this]() { run(); });
}

void LUTRendererThread::stopThread(int timeoutMs) {
    {
        std::lock_guard<std::mutex> lock(wakeupMutex_);
        shouldExit_.store(true, std::memory_order_release);
    }
    wakeupCV_.notify_one();

    if (thread_.joinable()) {
        if (timeoutMs > 0) {
            // Use a timed join approach via async
            auto future = std::async(std::launch::async, [this]() {
                thread_.join();
            });

            if (future.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::timeout) {
                // Thread didn't finish in time - detach to avoid blocking forever
                thread_.detach();
            }
        } else {
            thread_.join();
        }
    }
}

void LUTRendererThread::enqueueJob(const RenderJob& job) {
    {
        std::lock_guard<std::mutex> lock(jobQueueMutex_);

        // Simple ring buffer implementation
        constexpr int kQueueSize = 8;
        const int currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const int nextWrite = (currentWrite + 1) % kQueueSize;

        // Check if queue is full (would overwrite unread data)
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            // Queue full - drop oldest job and overwrite
            readIndex_.store((readIndex_.load(std::memory_order_relaxed) + 1) % kQueueSize,
                           std::memory_order_release);
        }

        jobSlots_[currentWrite] = job;
        writeIndex_.store(nextWrite, std::memory_order_release);
    }
    wakeupCV_.notify_one();
}

void LUTRendererThread::run() {
    while (!shouldExit_.load(std::memory_order_acquire)) {
        // Wait for work with timeout
        {
            std::unique_lock<std::mutex> lock(wakeupMutex_);
            wakeupCV_.wait_for(lock, std::chrono::seconds(1), [this]() {
                return shouldExit_.load(std::memory_order_acquire) ||
                       readIndex_.load(std::memory_order_acquire) !=
                       writeIndex_.load(std::memory_order_acquire);
            });
        }

        if (shouldExit_.load(std::memory_order_acquire)) {
            break;
        }

        processJobs();
    }
}

void LUTRendererThread::processJobs() {
    RenderJob latestJob;
    bool hasJob = false;

    // Drain all jobs, keeping only the latest
    {
        std::lock_guard<std::mutex> lock(jobQueueMutex_);
        constexpr int kQueueSize = 8;

        while (readIndex_.load(std::memory_order_relaxed) !=
               writeIndex_.load(std::memory_order_relaxed)) {
            const int currentRead = readIndex_.load(std::memory_order_relaxed);
            latestJob = jobSlots_[currentRead];
            readIndex_.store((currentRead + 1) % kQueueSize, std::memory_order_release);
            hasJob = true;
        }
    }

    if (hasJob) {
        // Render DSP LUT only (16K samples)
        // Visualizer is handled separately by VisualizerUpdateTimer (60Hz direct model reads)
        const int targetIdx = workerTargetIndex_.load(std::memory_order_relaxed);
        if (lutBuffers_ != nullptr) {
            renderDSPLUT(latestJob, &lutBuffers_[targetIdx]);
        }
    }
}

namespace {
    double computeNormalizationScalar(
        const std::array<double, TABLE_SIZE>& baseLayer,
        const std::array<double, 41>& coefficients,
        const HarmonicLayer& harmonicLayer,
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
        tempLTF_->setBaseLayerValue(i, job.baseLayerData[i]);
    }
    tempLTF_->setHarmonicCoefficients(job.coefficients);
    tempLTF_->setSplineAnchors(job.splineAnchors);
    tempLTF_->setExtrapolationMode(job.extrapolationMode);
    tempLTF_->setRenderingMode(job.renderingMode);

    // Compute normalization scalar (used by Harmonic mode, ignored by Paint/Spline modes)
    const double normScalar = computeNormalizationScalar(
        job.baseLayerData,
        job.coefficients,
        static_cast<const HarmonicLayer&>(tempLTF_->getHarmonicLayer()),
        job.normalizationEnabled,
        job.paintStrokeActive,
        job.frozenNormalizationScalar
    );

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
        outputBuffer->data[i] = tempLTF_->evaluateForRendering(x, normScalar);
    }

    outputBuffer->version = job.version;
    outputBuffer->extrapolationMode = job.extrapolationMode;

    newLUTReady_.store(true, std::memory_order_release);
}

void LUTRendererThread::setLUTBuffersPointer(LUTBuffer* buffers) {
    lutBuffers_ = buffers;
}

// LUTRenderTimer Implementation (20Hz, DSP LUT only, guaranteed delivery)

LUTRenderTimer::LUTRenderTimer(LayeredTransferFunction& ltf,
                               LUTRendererThread& renderer)
    : ltf_(ltf)
    , renderer_(renderer) {
    assert(platform::MessageThread::isThisTheMessageThread());
    startTimerHz(SeamlessConfig::DSP_TIMER_HZ);  // 50ms interval, matches crossfade timing
}

LUTRenderTimer::~LUTRenderTimer() {
    stopTimer();
}

void LUTRenderTimer::timerCallback() {
    assert(platform::MessageThread::isThisTheMessageThread());

    const uint64_t currentVersion = ltf_.getVersion();

    // Track version changes
    if (currentVersion != lastSeenVersion_) {
        lastSeenVersion_ = currentVersion;
    }

    // Render if we're behind - handles BOTH intermediate AND final renders
    // This two-version tracking guarantees the final change is never skipped
    if (lastRenderedVersion_ != lastSeenVersion_) {
        const RenderJob job = captureRenderJob();
        renderer_.enqueueJob(job);
        lastRenderedVersion_ = lastSeenVersion_;
    }
}

void LUTRenderTimer::forceRender() {
    assert(platform::MessageThread::isThisTheMessageThread());

    const RenderJob job = captureRenderJob();
    renderer_.enqueueJob(job);

    // Update both versions to prevent duplicate render on next tick
    const uint64_t currentVersion = ltf_.getVersion();
    lastSeenVersion_ = currentVersion;
    lastRenderedVersion_ = currentVersion;
}

RenderJob LUTRenderTimer::captureRenderJob() {
    RenderJob job;

    for (int i = 0; i < TABLE_SIZE; ++i) {
        job.baseLayerData[i] = ltf_.getBaseLayerValue(i);
    }

    job.coefficients = ltf_.getHarmonicCoefficients();
    job.splineAnchors = ltf_.getSplineLayer().getAnchors();
    job.normalizationEnabled = ltf_.isNormalizationEnabled();
    job.paintStrokeActive = ltf_.isPaintStrokeActive();
    job.renderingMode = ltf_.getRenderingMode();

    if (job.renderingMode == dsp_core::RenderingMode::Harmonic) {
        job.frozenNormalizationScalar = ltf_.getNormalizationScalar();
    } else {
        job.frozenNormalizationScalar = 1.0;
    }

    job.extrapolationMode = ltf_.getExtrapolationMode();
    job.version = ltf_.getVersion();

    return job;
}

// VisualizerUpdateTimer Implementation (120Hz, direct model reads)

VisualizerUpdateTimer::VisualizerUpdateTimer(LayeredTransferFunction& model)
    : editingModel_(model) {
    assert(platform::MessageThread::isThisTheMessageThread());
    startTimerHz(SeamlessConfig::VISUALIZER_TIMER_HZ);  // 120Hz for smooth UI updates during drag
}

VisualizerUpdateTimer::~VisualizerUpdateTimer() {
    stopTimer();
}

void VisualizerUpdateTimer::setVisualizerTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr,
                                                 std::function<void()> callback) {
    visualizerLUTPtr_ = lutPtr;
    onVisualizerUpdate_ = std::move(callback);
}

void VisualizerUpdateTimer::timerCallback() {
    assert(platform::MessageThread::isThisTheMessageThread());

    const uint64_t currentVersion = editingModel_.getVersion();

    // Path 1: Conditionally update transfer function curve (only when version changes)
    if (currentVersion != lastSeenVersion_) {
        lastSeenVersion_ = currentVersion;

        if (visualizerLUTPtr_) {
            // Update normalization scalar for Harmonic mode (must be computed fresh when harmonics change)
            // Paint mode: uses frozen scalar during strokes (controller manages this)
            // Spline mode: doesn't use normalization scalar
            if (editingModel_.getRenderingMode() == RenderingMode::Harmonic &&
                editingModel_.isNormalizationEnabled() &&
                !editingModel_.isPaintStrokeActive()) {
                editingModel_.updateNormalizationScalar();
            }

            // Get normalization scalar for rendering
            const double normScalar = editingModel_.getNormalizationScalar();

            // Direct model sampling - ~0.5ms for 1024 points
            for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
                const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1))
                                           * (MAX_VALUE - MIN_VALUE);
                (*visualizerLUTPtr_)[i] = editingModel_.evaluateForRendering(x, normScalar);
            }
        }
    }

    // Path 2: Unconditionally invoke callback (for amplitude trace updates)
    // The callback handles both curve updates (when LUT changed) and amplitude trace (every frame)
    if (onVisualizerUpdate_) {
        onVisualizerUpdate_();
    }
}

void VisualizerUpdateTimer::forceUpdate() {
    assert(platform::MessageThread::isThisTheMessageThread());

    if (visualizerLUTPtr_) {
        // Update normalization scalar for Harmonic mode (must be computed fresh when harmonics change)
        if (editingModel_.getRenderingMode() == RenderingMode::Harmonic &&
            editingModel_.isNormalizationEnabled() &&
            !editingModel_.isPaintStrokeActive()) {
            editingModel_.updateNormalizationScalar();
        }

        // Get normalization scalar for rendering
        const double normScalar = editingModel_.getNormalizationScalar();

        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1))
                                       * (MAX_VALUE - MIN_VALUE);
            (*visualizerLUTPtr_)[i] = editingModel_.evaluateForRendering(x, normScalar);
        }

        if (onVisualizerUpdate_) {
            onVisualizerUpdate_();
        }
    }

    lastSeenVersion_ = editingModel_.getVersion();
}

} // namespace dsp_core
