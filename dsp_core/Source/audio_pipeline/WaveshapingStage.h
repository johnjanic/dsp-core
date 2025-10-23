#pragma once

#include "AudioProcessingStage.h"
#include "../LayeredTransferFunction.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

/**
 * Applies waveshaping using LayeredTransferFunction.
 * Supports parallel processing via injected thread pool.
 */
class WaveshapingStage : public AudioProcessingStage {
public:
    /**
     * @param ltf Reference to transfer function model
     * @param threadPool Thread pool for parallel processing (nullptr = serial)
     */
    WaveshapingStage(dsp_core::LayeredTransferFunction& ltf, juce::ThreadPool* threadPool = nullptr);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "Waveshaping"; }

private:
    void processParallel(juce::AudioBuffer<double>& buffer);
    void processSerial(juce::AudioBuffer<double>& buffer);

    dsp_core::LayeredTransferFunction& ltf_;
    juce::ThreadPool* threadPool_;  // Not owned (injected dependency)
};

} // namespace dsp_core::audio_pipeline
