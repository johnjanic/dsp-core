#pragma once

#include "AudioProcessingStage.h"
#include "../engine/SeamlessTransferFunction.h"
#include "../model/LayeredTransferFunction.h"

namespace dsp_core::audio_pipeline {

/**
 * Applies waveshaping using SeamlessTransferFunction or LayeredTransferFunction.
 * Optimized for stereo processing (no threading overhead).
 */
class WaveshapingStage : public AudioProcessingStage {
  public:
    /**
     * @param tf Reference to seamless transfer function (production use)
     */
    explicit WaveshapingStage(const dsp_core::SeamlessTransferFunction& tf);

    /**
     * @param ltf Reference to layered transfer function (testing/profiling use)
     */
    explicit WaveshapingStage(dsp_core::LayeredTransferFunction& ltf);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Waveshaping";
    }

  private:
    const dsp_core::SeamlessTransferFunction* seamlessTransferFunction_{nullptr};
    dsp_core::LayeredTransferFunction* layeredTransferFunction_{nullptr};
};

} // namespace dsp_core::audio_pipeline
