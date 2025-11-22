#include "WaveshapingStage.h"

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf) : ltf_(ltf) {}

void WaveshapingStage::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/) {
    // Waveshaping is stateless, no preparation needed
}

void WaveshapingStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        double* channelData = buffer.getWritePointer(ch);
        ltf_.processBlock(channelData, numSamples);
    }
}

void WaveshapingStage::reset() {
    // Waveshaping is stateless
}

} // namespace dsp_core::audio_pipeline
