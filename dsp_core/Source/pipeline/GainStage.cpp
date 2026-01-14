#include "GainStage.h"

namespace dsp_core::audio_pipeline {

void GainStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    gain_.prepare(sampleRate);
}

void GainStage::process(platform::AudioBuffer<double>& buffer) {
    gain_.processBlock(buffer);
}

void GainStage::reset() {
    gain_.reset();
}

void GainStage::setGainDecibels(double gainDB) {
    gain_.setGainDecibels(gainDB);
}

void GainStage::setGainLinear(double gain) {
    gain_.setGainLinear(gain);
}

double GainStage::getGainDecibels() const {
    return gain_.getGainDecibels();
}

double GainStage::getGainLinear() const {
    return gain_.getGainLinear();
}

} // namespace dsp_core::audio_pipeline
