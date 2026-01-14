#include "DCBlockingFilter.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// Cutoff frequency limits (safety: preserve low-end, avoid audible filtering)
constexpr double kMinFrequencyHz = 1.0;
constexpr double kMaxFrequencyHz = 20.0;
constexpr double kPi = 3.14159265358979323846;
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

    // Design 1st-order Butterworth highpass filter using bilinear transform
    // Analog prototype: H(s) = s / (s + ωc)
    // Bilinear transform: s = 2*fs * (1 - z^-1) / (1 + z^-1)
    const double wc = 2.0 * kPi * cutoffHz;
    const double k = std::tan(wc / (2.0 * sampleRate_));

    // Coefficients for highpass: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    const double norm = 1.0 / (1.0 + k);
    const double b0 = norm;
    const double b1 = -norm;
    const double a1 = (k - 1.0) * norm;

    // Update all filters
    for (auto& filter : filters_) {
        filter.b0 = b0;
        filter.b1 = b1;
        filter.a1 = a1;
    }
}

} // namespace dsp_core::audio_pipeline
