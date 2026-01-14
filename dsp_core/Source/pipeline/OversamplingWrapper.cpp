#include "OversamplingWrapper.h"
#include <cassert>

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxOversamplingOrder = 4;
constexpr int kNumOversamplingModes = kMaxOversamplingOrder + 1; // Orders 0-4
constexpr int kNumChannels = 2; // Stereo
} // namespace

OversamplingWrapper::OversamplingWrapper(std::unique_ptr<AudioProcessingStage> wrappedStage, int oversamplingOrder)
    : wrappedStage_(std::move(wrappedStage)), currentOrder_(oversamplingOrder) {
    assert(wrappedStage_ != nullptr);
    assert(oversamplingOrder >= 0 && oversamplingOrder <= kMaxOversamplingOrder);

    // Pre-create all oversamplers with dsp::Oversampling
    for (int i = 0; i < kNumOversamplingModes; ++i) {
        oversamplers_[i] = std::make_unique<dsp::Oversampling<double>>(kNumChannels, i);
    }
}

void OversamplingWrapper::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    // Prepare all oversamplers
    for (auto& oversampler : oversamplers_) {
        oversampler->prepare(samplesPerBlock);
        oversampler->reset();
    }

    // Prepare wrapped stage at oversampled rate
    const int factor = 1 << currentOrder_; // 2^order
    const int oversampledBlockSize = samplesPerBlock * factor;
    wrappedStage_->prepareToPlay(sampleRate * factor, oversampledBlockSize);
}

void OversamplingWrapper::process(audio::AudioBuffer<double>& buffer) {
    auto& oversampler = *oversamplers_[currentOrder_];

    // 1. Upsample - returns reference to internal oversampled buffer
    auto& oversampledBuffer = oversampler.processSamplesUp(buffer);

    // 2. Process wrapped stage at high sample rate
    wrappedStage_->process(oversampledBuffer);

    // 3. Downsample back to original rate
    oversampler.processSamplesDown(buffer);
}

void OversamplingWrapper::reset() {
    for (auto& oversampler : oversamplers_) {
        oversampler->reset();
    }
    wrappedStage_->reset();
}

std::string OversamplingWrapper::getName() const {
    const int factor = 1 << currentOrder_;
    return std::to_string(factor) + "x(" + wrappedStage_->getName() + ")";
}

int OversamplingWrapper::getLatencySamples() const {
    const auto& oversampler = *oversamplers_[currentOrder_];
    const int oversamplingLatency = oversampler.getLatencyInSamples();
    const int wrappedLatency = wrappedStage_->getLatencySamples();

    // Wrapped latency is at oversampled rate, convert to base rate
    const int factor = 1 << currentOrder_;
    return oversamplingLatency + (wrappedLatency / factor);
}

void OversamplingWrapper::setOversamplingOrder(int order) {
    assert(order >= 0 && order <= kMaxOversamplingOrder);
    if (order == currentOrder_) {
        return;
    }

    currentOrder_ = order;

    // Re-prepare wrapped stage at new sample rate
    const int factor = 1 << currentOrder_;
    const int oversampledBlockSize = maxBlockSize_ * factor;
    wrappedStage_->prepareToPlay(sampleRate_ * factor, oversampledBlockSize);
}

} // namespace dsp_core::audio_pipeline
