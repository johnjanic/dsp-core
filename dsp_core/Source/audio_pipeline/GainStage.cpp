#include "GainStage.h"

namespace dsp_core::audio_pipeline {

void GainStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    smoothedGain_.reset(sampleRate, 0.01);  // 10ms ramp time
    isPrepared_ = true;
}

void GainStage::process(juce::AudioBuffer<double>& buffer) {
    if (!isPrepared_) {
        jassertfalse;  // Must call prepareToPlay() first
        return;
    }

    // Prepare spec dynamically based on actual buffer
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate_;
    spec.maximumBlockSize = static_cast<juce::uint32>(buffer.getNumSamples());
    spec.numChannels = static_cast<juce::uint32>(buffer.getNumChannels());

    gainProcessor_.prepare(spec);
    gainProcessor_.setGainLinear(smoothedGain_.getNextValue());

    // Process buffer
    juce::dsp::AudioBlock<double> block(buffer);
    juce::dsp::ProcessContextReplacing<double> context(block);
    gainProcessor_.process(context);

    // Skip smoothing for remaining samples
    if (buffer.getNumSamples() > 1) {
        smoothedGain_.skip(buffer.getNumSamples() - 1);
    }
}

void GainStage::reset() {
    gainProcessor_.reset();
    smoothedGain_.setCurrentAndTargetValue(smoothedGain_.getTargetValue());
}

void GainStage::setGainDB(double gainDB) {
    double linear = juce::Decibels::decibelsToGain(gainDB);
    smoothedGain_.setTargetValue(linear);
}

void GainStage::setGainLinear(double gainLinear) {
    smoothedGain_.setTargetValue(gainLinear);
}

double GainStage::getTargetGainLinear() const {
    return smoothedGain_.getTargetValue();
}

} // namespace dsp_core::audio_pipeline
