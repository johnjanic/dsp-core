#include "DCBlockingFilter.h"
#include <algorithm>

namespace dsp_core::audio_pipeline {

namespace {
// Cutoff frequency limits (safety: preserve low-end, avoid audible filtering)
constexpr double kMinFrequencyHz = 1.0;
constexpr double kMaxFrequencyHz = 20.0;
} // namespace

void DCBlockingFilter::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;

    // Resize filters for stereo (will be resized in process() if needed)
    const int numChannels = 2;
    filters_.resize(numChannels);

    // Configure all filter coefficients
    updateFilterCoefficients();
}

void DCBlockingFilter::process(platform::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Resize filters if channel count changed
    if (filters_.size() != static_cast<size_t>(numChannels)) {
        filters_.resize(numChannels);
        updateFilterCoefficients();
    }

    // Early return if disabled
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Process each channel: apply DC blocking filter
    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            data[i] = filters_[ch].processSample(data[i]);
        }
    }
}

void DCBlockingFilter::reset() {
    // Reset all filter states
    for (auto& filter : filters_) {
        filter.reset();
    }
}

void DCBlockingFilter::setCutoffFrequency(double frequencyHz) {
    // Clamp to 1-20Hz range (safety: preserve low-end, avoid audible filtering)
    frequencyHz = std::clamp(frequencyHz, kMinFrequencyHz, kMaxFrequencyHz);

    // Store with atomic release
    cutoffFrequency_.store(frequencyHz, std::memory_order_release);

    // Update filter coefficients (UI thread only)
    updateFilterCoefficients();
}

void DCBlockingFilter::updateFilterCoefficients() {
    // Load cutoff frequency
    const double cutoffHz = cutoffFrequency_.load(std::memory_order_acquire);

    // Create first-order highpass coefficients using dsp::IIRCoefficients
    auto coeffs = dsp::IIRCoefficients<double>::makeFirstOrderHighPass(sampleRate_, cutoffHz);

    // Update all filters
    for (auto& filter : filters_) {
        filter.setCoefficients(coeffs);
    }
}

} // namespace dsp_core::audio_pipeline
