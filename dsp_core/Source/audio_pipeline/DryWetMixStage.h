#pragma once

#include "AudioProcessingStage.h"
#include <memory>

namespace dsp_core::audio_pipeline {

/**
 * Wrapper stage that applies dry/wet mixing to wrapped stage(s).
 *
 * Usage:
 *   auto waveshaper = std::make_unique<WaveshapingStage>(ltf);
 *   auto mixed = std::make_unique<DryWetMixStage>(std::move(waveshaper));
 *   mixed->setMixAmount(0.5);  // 50% dry, 50% wet
 *   mixed->process(buffer);
 *
 * Can wrap single stage or entire sub-pipeline.
 */
class DryWetMixStage : public AudioProcessingStage {
public:
    /**
     * Wrap a single stage with dry/wet mixing.
     */
    explicit DryWetMixStage(std::unique_ptr<AudioProcessingStage> wrappedStage);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;
    int getLatencySamples() const override;

    /**
     * Set mix amount.
     * @param mix 0.0 = 100% dry (bypass), 1.0 = 100% wet (full effect)
     */
    void setMixAmount(double mix);

    /**
     * Get current mix amount.
     */
    double getMixAmount() const { return mixAmount_; }

private:
    void captureDrySignal(const juce::AudioBuffer<double>& buffer);
    void applyMix(juce::AudioBuffer<double>& wetBuffer);

    std::unique_ptr<AudioProcessingStage> wrappedStage_;
    juce::AudioBuffer<double> dryBuffer_;
    double mixAmount_ = 1.0;  // 100% wet by default
};

} // namespace dsp_core::audio_pipeline
