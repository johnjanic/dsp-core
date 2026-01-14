#include "OversamplingWrapper.h"
#include <cassert>

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxOversamplingOrder = 4;
constexpr int kNumOversamplingModes = kMaxOversamplingOrder + 1; // Orders 0-4
} // namespace

OversamplingWrapper::OversamplingWrapper(std::unique_ptr<AudioProcessingStage> wrappedStage, int oversamplingOrder)
    : wrappedStage_(std::move(wrappedStage)), currentOrder_(oversamplingOrder) {
    assert(wrappedStage_ != nullptr);
    assert(oversamplingOrder >= 0 && oversamplingOrder <= kMaxOversamplingOrder);

    // Pre-create all oversamplers
    for (int i = 0; i < kNumOversamplingModes; ++i) {
        oversamplers_[i] = std::make_unique<juce::dsp::Oversampling<double>>(
            2, // 2 channels (stereo)
            i, // Oversampling order
            juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR);
    }
}

void OversamplingWrapper::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    // Prepare all oversamplers
    juce::dsp::ProcessSpec spec{};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;

    for (auto& oversampler : oversamplers_) {
        oversampler->initProcessing(samplesPerBlock);
        oversampler->reset();
    }

    // Prepare wrapped stage at oversampled rate
    const int factor = 1 << currentOrder_; // 2^order
    const int oversampledBlockSize = samplesPerBlock * factor;
    wrappedStage_->prepareToPlay(sampleRate * factor, oversampledBlockSize);
}

void OversamplingWrapper::process(platform::AudioBuffer<double>& buffer) {
    auto& oversampler = *oversamplers_[currentOrder_];
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Create JUCE AudioBlock from platform::AudioBuffer data
    // NOTE: juce::dsp::AudioBlock can work with raw pointers
    juce::dsp::AudioBlock<double> block(buffer.getArrayOfWritePointers(),
                                        static_cast<size_t>(numChannels),
                                        static_cast<size_t>(numSamples));

    // 1. Upsample
    auto oversampledBlock = oversampler.processSamplesUp(block);

    // 2. Create platform::AudioBuffer wrapper for oversampled data
    // PERFORMANCE FIX: Use pre-allocated array instead of std::vector to avoid allocation
    const size_t oversampledNumChannels = oversampledBlock.getNumChannels();
    assert(oversampledNumChannels <= channelPointers_.size());

    for (size_t ch = 0; ch < oversampledNumChannels; ++ch) {
        channelPointers_[ch] = oversampledBlock.getChannelPointer(ch);
    }

    // Create a platform::AudioBuffer that wraps the oversampled data
    // Note: We need to process in-place, so we create a temporary buffer
    platform::AudioBuffer<double> oversampledBuffer(static_cast<int>(oversampledNumChannels),
                                                    static_cast<int>(oversampledBlock.getNumSamples()));
    for (size_t ch = 0; ch < oversampledNumChannels; ++ch) {
        double* dest = oversampledBuffer.getWritePointer(static_cast<int>(ch));
        const double* src = oversampledBlock.getChannelPointer(ch);
        for (size_t i = 0; i < oversampledBlock.getNumSamples(); ++i) {
            dest[i] = src[i];
        }
    }

    // 3. Process wrapped stage at high sample rate
    wrappedStage_->process(oversampledBuffer);

    // 4. Copy processed data back to oversampled block for downsampling
    for (size_t ch = 0; ch < oversampledNumChannels; ++ch) {
        double* dest = oversampledBlock.getChannelPointer(ch);
        const double* src = oversampledBuffer.getReadPointer(static_cast<int>(ch));
        for (size_t i = 0; i < oversampledBlock.getNumSamples(); ++i) {
            dest[i] = src[i];
        }
    }

    // 5. Downsample
    oversampler.processSamplesDown(block);

    // 6. Copy downsampled data back to platform buffer (block writes to same pointers)
    // The AudioBlock was constructed with buffer's pointers, so data is already in place
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
    const int oversamplingLatency = static_cast<int>(oversampler.getLatencyInSamples());
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
