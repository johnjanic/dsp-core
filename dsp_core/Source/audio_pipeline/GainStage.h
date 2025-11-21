#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>

namespace dsp_core::audio_pipeline {

/**
 * Applies gain with smoothing to prevent clicks.
 *
 * Features:
 * - Automatic channel count detection
 * - 10ms smoothing time for parameter changes
 * - Supports both dB and linear gain setting
 */
class GainStage : public AudioProcessingStage {
  public:
    GainStage() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Gain";
    }

    /**
     * Set gain in decibels.
     * Smoothed over ~10ms to prevent clicks.
     */
    void setGainDB(double gainDB);

    /**
     * Set gain as linear multiplier.
     */
    void setGainLinear(double gainLinear);

    /**
     * Get current target gain (linear).
     */
    double getTargetGainLinear() const;

  private:
    juce::dsp::Gain<double> gainProcessor_;
    juce::LinearSmoothedValue<double> smoothedGain_{1.0};
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
    int numChannels_ = 2;
    bool isPrepared_ = false;
};

} // namespace dsp_core::audio_pipeline
