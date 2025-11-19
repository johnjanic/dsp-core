#include "SilenceDetector.h"

namespace dsp_core::audio_pipeline {

void SilenceDetector::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;

    // Initialize per-channel envelopes (typically 2 for stereo)
    // Note: Buffer channel count determined at process() time
    channelEnvelopes_.clear();
    channelEnvelopes_.resize(2); // Default to stereo, will resize if needed

    for (auto& envelope : channelEnvelopes_) {
        envelope.configure(sampleRate, 20.0); // 20ms decay time (balanced: fast but stable)
    }

    // Start with "not silent" to avoid false positives on startup
    isNearSilence_.store(false, std::memory_order_release);
}

void SilenceDetector::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Resize envelope detectors if channel count changed
    if (static_cast<int>(channelEnvelopes_.size()) != numChannels) {
        channelEnvelopes_.resize(numChannels);
        for (auto& envelope : channelEnvelopes_) {
            envelope.configure(sampleRate_, 20.0); // 20ms decay time (balanced: fast but stable)
        }
    }

    // Update per-channel envelopes
    for (int ch = 0; ch < numChannels; ++ch) {
        const double* data = buffer.getReadPointer(ch);
        auto& envelope = channelEnvelopes_[ch];

        for (int i = 0; i < numSamples; ++i) {
            envelope.process(data[i]);
        }
    }

    // Determine if ALL channels are near silence (logical AND)
    bool allChannelsSilent = true;
    for (const auto& envelope : channelEnvelopes_) {
        if (!envelope.isNearSilence()) {
            allChannelsSilent = false;
            break;
        }
    }

    // Store result (atomic write, read by DynamicOutputBiasing)
    isNearSilence_.store(allChannelsSilent, std::memory_order_release);

    // Pass through buffer unmodified (detection only, no processing)
}

void SilenceDetector::reset() {
    // Clear all envelope state
    for (auto& envelope : channelEnvelopes_) {
        envelope.reset();
    }

    // Reset to "not silent"
    isNearSilence_.store(false, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
