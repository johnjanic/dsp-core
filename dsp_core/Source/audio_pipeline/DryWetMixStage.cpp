#include "DryWetMixStage.h"

namespace dsp_core::audio_pipeline {

DryWetMixStage::DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline)
    : effectsPipeline_(std::move(effectsPipeline))
{
    jassert(effectsPipeline_ != nullptr);
}

void DryWetMixStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Allocate dry buffer for worst-case
    // Assume max 8 channels (stereo, 5.1, 7.1)
    dryBuffer_.setSize(8, samplesPerBlock, false, true, true);

    // Prepare effects pipeline
    effectsPipeline_->prepareToPlay(sampleRate, samplesPerBlock);
}

void DryWetMixStage::process(juce::AudioBuffer<double>& buffer) {
    // 1. Capture dry signal (before any processing)
    captureDrySignal(buffer);

    // 2. Process wet signal through effects pipeline (includes gain stages)
    effectsPipeline_->process(buffer);

    // 3. Mix dry and wet
    applyMix(buffer);
}

void DryWetMixStage::reset() {
    effectsPipeline_->reset();
    dryBuffer_.clear();
}

juce::String DryWetMixStage::getName() const {
    return "DryWetMix(" + effectsPipeline_->getName() + ")";
}

int DryWetMixStage::getLatencySamples() const {
    return effectsPipeline_->getLatencySamples();
}

AudioPipeline* DryWetMixStage::getEffectsPipeline() {
    return effectsPipeline_.get();
}

void DryWetMixStage::setMixAmount(double mix) {
    mixAmount_ = juce::jlimit(0.0, 1.0, mix);
}

void DryWetMixStage::captureDrySignal(const juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Safety checks (should never fail if prepareToPlay called correctly)
    jassert(numChannels <= dryBuffer_.getNumChannels());
    jassert(numSamples <= dryBuffer_.getNumSamples());

    // Copy dry signal (no allocation)
    for (int ch = 0; ch < numChannels; ++ch) {
        dryBuffer_.copyFrom(ch, 0, buffer, ch, 0, numSamples);
    }
}

void DryWetMixStage::applyMix(juce::AudioBuffer<double>& wetBuffer) {
    const int numChannels = wetBuffer.getNumChannels();
    const int numSamples = wetBuffer.getNumSamples();

    const double dryGain = 1.0 - mixAmount_;
    const double wetGain = mixAmount_;

    for (int ch = 0; ch < numChannels; ++ch) {
        double* wetData = wetBuffer.getWritePointer(ch);
        const double* dryData = dryBuffer_.getReadPointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            wetData[i] = dryData[i] * dryGain + wetData[i] * wetGain;
        }
    }
}

} // namespace dsp_core::audio_pipeline
