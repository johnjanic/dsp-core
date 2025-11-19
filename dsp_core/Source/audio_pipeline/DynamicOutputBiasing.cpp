#include "DynamicOutputBiasing.h"

namespace dsp_core::audio_pipeline {

DynamicOutputBiasing::DynamicOutputBiasing(LayeredTransferFunction& ltf, SilenceDetector& silenceDetector)
    : ltf_(ltf), silenceDetector_(silenceDetector),
      lastBiasUpdate_() // Default-constructed (epoch), allows first update immediately
{}

void DynamicOutputBiasing::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;

    // Configure fade controller
    fade_.configure(sampleRate);

    // Compute initial bias
    updateBias();
}

void DynamicOutputBiasing::process(juce::AudioBuffer<double>& buffer) {
    // Bypass if disabled
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Read bias once per buffer (minimize atomic loads)
    const double bias = cachedBias_.load(std::memory_order_acquire);

    // Read silence state from detector (computed by previous stage)
    const bool nearSilence = silenceDetector_.isNearSilence();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Update fade state once per sample (call process() outside channel loop)
    // Process samples (fade updates once per sample, applied to all channels)
    for (int i = 0; i < numSamples; ++i) {
        // Update fade controller (once per sample)
        fade_.process(nearSilence);
        double N = fade_.getNextValue();

        // Apply bias to all channels
        for (int ch = 0; ch < numChannels; ++ch) {
            double* data = buffer.getWritePointer(ch);
            data[i] -= (N * bias);
        }
    }
}

void DynamicOutputBiasing::reset() {
    fade_.reset();
}

void DynamicOutputBiasing::notifyTransferFunctionChanged() {
    updateBias();
}

void DynamicOutputBiasing::updateBias() {
    // Rate limiting: max 20 updates/sec (avoid excessive evaluations during rapid editing)
    auto now = std::chrono::steady_clock::now();
    if (now - lastBiasUpdate_ < std::chrono::milliseconds(kDebounceMs)) {
        return; // Debounce
    }
    lastBiasUpdate_ = now;

    // Compute DC offset: B = f(0)
    // This is trivial for output biasing (just evaluate at x=0)
    double dcOffset = ltf_.applyTransferFunction(0.0);

    // Atomic write (UI thread → audio thread)
    cachedBias_.store(dcOffset, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
