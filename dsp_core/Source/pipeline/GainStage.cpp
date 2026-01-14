#include "GainStage.h"
#include <cassert>
#include <cmath>

namespace dsp_core::audio_pipeline {

void GainStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    // Reset smoothing state
    currentGain_ = targetGain_;
    samplesRemaining_ = 0;
    gainStep_ = 0.0;

    isPrepared_ = true;
}

void GainStage::process(platform::AudioBuffer<double>& buffer) {
    assert(isPrepared_); // Must call prepareToPlay() first

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    if (numSamples == 0 || numChannels == 0) {
        return;
    }

    // Check if we need smoothing
    if (samplesRemaining_ > 0) {
        // Apply gain with smoothing
        for (int sample = 0; sample < numSamples; ++sample) {
            if (samplesRemaining_ > 0) {
                currentGain_ += gainStep_;
                --samplesRemaining_;
            }

            for (int ch = 0; ch < numChannels; ++ch) {
                double* data = buffer.getWritePointer(ch);
                data[sample] *= currentGain_;
            }
        }

        // Snap to target when done
        if (samplesRemaining_ <= 0) {
            currentGain_ = targetGain_;
        }
    } else if (currentGain_ != 1.0) {
        // Apply constant gain (no smoothing needed)
        buffer.applyGain(currentGain_);
    }
    // If gain is 1.0 and no smoothing, do nothing (unity gain)
}

void GainStage::reset() {
    currentGain_ = targetGain_;
    samplesRemaining_ = 0;
    gainStep_ = 0.0;
}

void GainStage::setGainDB(double gainDB) {
    // Convert dB to linear: 10^(dB/20)
    const double linear = std::pow(10.0, gainDB / 20.0);
    targetGain_ = linear;
    updateSmoothing();
}

void GainStage::updateSmoothing() {
    if (currentGain_ != targetGain_ && isPrepared_) {
        samplesRemaining_ = static_cast<int>(sampleRate_ * kGainRampTimeSeconds);
        if (samplesRemaining_ > 0) {
            gainStep_ = (targetGain_ - currentGain_) / static_cast<double>(samplesRemaining_);
        } else {
            currentGain_ = targetGain_;
            gainStep_ = 0.0;
        }
    }
}

} // namespace dsp_core::audio_pipeline
