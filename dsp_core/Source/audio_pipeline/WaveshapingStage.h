#pragma once

#include "AudioProcessingStage.h"
#include "../LayeredTransferFunction.h"

namespace dsp_core::audio_pipeline {

/**
 * Applies waveshaping using LayeredTransferFunction.
 * Optimized for stereo processing (no threading overhead).
 */
class WaveshapingStage : public AudioProcessingStage {
public:
    /**
     * @param ltf Reference to transfer function model
     */
    explicit WaveshapingStage(dsp_core::LayeredTransferFunction& ltf);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "Waveshaping"; }

private:
    dsp_core::LayeredTransferFunction& ltf_;
};

} // namespace dsp_core::audio_pipeline
