#pragma once

#include "AudioProcessingStage.h"
#include "PeakEnvelopeDetector.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * Lightweight silence detection stage for dynamic DC compensation.
 *
 * Runs BEFORE waveshaping to detect when input signal is near silence.
 * Stores detection result in atomic flag for downstream stages to read.
 *
 * Architecture:
 * - Per-channel peak envelope detection
 * - Logical AND across all channels (all must be silent)
 * - Zero latency (instant attack on transients)
 *
 * Pipeline Position: Before WaveshapingStage (inside oversampling wrapper)
 *
 * Thread Safety:
 * - Audio thread: Updates envelopes, writes atomic flag
 * - Other stages (audio thread): Read atomic flag
 *
 * CPU Cost: ~4 FLOPs per sample (envelope detection only)
 */
class SilenceDetector : public AudioProcessingStage {
  public:
    SilenceDetector() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "SilenceDetector";
    }

    /**
     * Check if signal is currently near silence.
     *
     * Thread-safe: uses atomic load.
     * Returns true only if ALL channels are below silence threshold.
     *
     * @return true if all channels are near silence (-60 dBFS)
     */
    bool isNearSilence() const {
        return isNearSilence_.load(std::memory_order_acquire);
    }

    /**
     * Get peak level for specific channel (for debugging/visualization).
     *
     * @param channel Channel index
     * @return Current peak level [0.0, 1.0+], or 0.0 if channel invalid
     */
    double getPeakLevel(int channel) const {
        if (channel >= 0 && channel < static_cast<int>(channelEnvelopes_.size())) {
            return channelEnvelopes_[channel].getPeakLevel();
        }
        return 0.0;
    }

  private:
    std::vector<PeakEnvelopeDetector> channelEnvelopes_;
    std::atomic<bool> isNearSilence_{false};
    double sampleRate_ = 44100.0;
};

} // namespace dsp_core::audio_pipeline
