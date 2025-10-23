#include "AudioPipeline.h"

namespace dsp_core::audio_pipeline {

void AudioPipeline::addStage(std::unique_ptr<AudioProcessingStage> stage) {
    jassert(stage != nullptr);
    stages_.push_back(std::move(stage));
}

void AudioPipeline::prepareToPlay(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    for (auto& stage : stages_) {
        stage->prepareToPlay(sampleRate, samplesPerBlock);
    }
}

void AudioPipeline::process(juce::AudioBuffer<double>& buffer) {
    for (auto& stage : stages_) {
        stage->process(buffer);
    }
}

void AudioPipeline::reset() {
    for (auto& stage : stages_) {
        stage->reset();
    }
}

int AudioPipeline::getTotalLatencySamples() const {
    int totalLatency = 0;
    for (const auto& stage : stages_) {
        totalLatency += stage->getLatencySamples();
    }
    return totalLatency;
}

} // namespace dsp_core::audio_pipeline
