#pragma once

#include "AudioProcessingStage.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Applies gain with smoothing to prevent clicks.
 *
 * Features:
 * - Automatic channel count detection
 * - 10ms smoothing time for parameter changes
 * - dB-based gain control
 */
class GainStage : public AudioProcessingStage {
  public:
    GainStage() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(platform::AudioBuffer<double>& buffer) override;
    void reset() override;
    std::string getName() const override {
        return "Gain";
    }

    /**
     * Set gain in decibels.
     * Smoothed over ~10ms to prevent clicks.
     */
    void setGainDB(double gainDB);

  private:
    // Linear smoothing state
    double currentGain_ = 1.0;
    double targetGain_ = 1.0;
    double gainStep_ = 0.0;
    int samplesRemaining_ = 0;

    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
    bool isPrepared_ = false;

    static constexpr double kGainRampTimeSeconds = 0.01; // 10ms ramp time

    void updateSmoothing();
};

} // namespace dsp_core::audio_pipeline
