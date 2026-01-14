#include "WaveshapingStage.h"

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(const dsp_core::SeamlessTransferFunction& tf)
    : seamlessTransferFunction_(&tf) {}

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf)
    : layeredTransferFunction_(&ltf) {}

void WaveshapingStage::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/) {
    // Waveshaping is stateless, no preparation needed
}

void WaveshapingStage::process(platform::AudioBuffer<double>& buffer) {
    if (seamlessTransferFunction_ != nullptr) {
        // Use new multi-channel processBuffer() API for correct crossfade handling
        seamlessTransferFunction_->processBuffer(buffer);
    } else if (layeredTransferFunction_ != nullptr) {
        // LayeredTransferFunction still uses per-channel processing
        const int numChannels = buffer.getNumChannels();
        const int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < numChannels; ++ch) {
            double* channelData = buffer.getWritePointer(ch);
            layeredTransferFunction_->processBlock(channelData, numSamples);
        }
    }
}

void WaveshapingStage::reset() {
    // Waveshaping is stateless
}

} // namespace dsp_core::audio_pipeline
