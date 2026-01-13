#include "GainStage.h"

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kGainRampTimeSeconds = 0.01; // 10ms ramp time
} // namespace

void GainStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    smoothedGain_.reset(sampleRate, kGainRampTimeSeconds);

    // CRITICAL: Prepare gainProcessor ONCE in prepareToPlay, NOT in process()
    // Calling prepare() in audio thread causes allocations!
    juce::dsp::ProcessSpec spec{};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = static_cast<juce::uint32>(numChannels_);
    gainProcessor_.prepare(spec);

    isPrepared_ = true;
}

void GainStage::process(juce::AudioBuffer<double>& buffer) {
    if (!isPrepared_) {
        jassertfalse; // Must call prepareToPlay() first
        return;
    }

    // PERFORMANCE FIX: Don't call prepare() here - it allocates memory!
    // Just set gain and process (like old code)
    gainProcessor_.setGainLinear(smoothedGain_.getNextValue());

    // Process buffer
    juce::dsp::AudioBlock<double> block(buffer);
    const juce::dsp::ProcessContextReplacing<double> context(block);
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
    const double linear = juce::Decibels::decibelsToGain(gainDB);
    smoothedGain_.setTargetValue(linear);
}

} // namespace dsp_core::audio_pipeline
