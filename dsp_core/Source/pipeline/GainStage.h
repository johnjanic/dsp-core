#pragma once

#include "AudioProcessingStage.h"
#include "../primitives/Gain.h"

namespace dsp_core::audio_pipeline {

/**
 * Applies gain with smoothing to prevent clicks.
 *
 * Features:
 * - Automatic channel count detection
 * - 10ms smoothing time for parameter changes
 * - dB and linear gain control
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
    void setGainDecibels(double gainDB);

    /**
     * Set gain as linear multiplier.
     * Smoothed over ~10ms to prevent clicks.
     */
    void setGainLinear(double gain);

    /**
     * Get target gain in decibels.
     */
    [[nodiscard]] double getGainDecibels() const;

    /**
     * Get target gain as linear multiplier.
     */
    [[nodiscard]] double getGainLinear() const;

    // Legacy alias for backward compatibility
    void setGainDB(double gainDB) { setGainDecibels(gainDB); }

  private:
    dsp::Gain<double> gain_;
};

} // namespace dsp_core::audio_pipeline
