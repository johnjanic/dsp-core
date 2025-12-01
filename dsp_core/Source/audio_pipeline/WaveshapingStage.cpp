#include "WaveshapingStage.h"

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(const dsp_core::SeamlessTransferFunction& tf)
    : seamlessTransferFunction_(&tf), layeredTransferFunction_(nullptr) {}

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf)
    : seamlessTransferFunction_(nullptr), layeredTransferFunction_(&ltf) {}

void WaveshapingStage::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/) {
    // Waveshaping is stateless, no preparation needed
}

void WaveshapingStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        double* channelData = buffer.getWritePointer(ch);

        if (seamlessTransferFunction_ != nullptr) {
            seamlessTransferFunction_->processBlock(channelData, numSamples);
        } else if (layeredTransferFunction_ != nullptr) {
            layeredTransferFunction_->processBlock(channelData, numSamples);
        }
    }
}

void WaveshapingStage::reset() {
    // Waveshaping is stateless
}

} // namespace dsp_core::audio_pipeline
