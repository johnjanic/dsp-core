#include "AudioPipelineBuilder.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

AudioPipelineBuilder::AudioPipelineBuilder()
    : pipeline_(std::make_unique<AudioPipeline>()) {}

AudioPipelineBuilder& AudioPipelineBuilder::withDryWetMix() {
    jassert(!built_ && "Cannot modify builder after build()");
    jassert(pipeline_->getNumStages() == 0 && "withDryWetMix() must be called before adding stages");
    useDryWetMix_ = true;
    return *this;
}

AudioPipelineBuilder::BuildResult AudioPipelineBuilder::build() {
    jassert(!built_ && "Builder already consumed");
    built_ = true;

    if (useDryWetMix_) {
        // Cache the inner pipeline pointer before wrapping
        auto* innerPipeline = pipeline_.get();
        handles_.setEffectsPipeline(innerPipeline);

        // Wrap in DryWetMixStage
        auto dryWet = std::make_unique<DryWetMixStage>(std::move(pipeline_));
        handles_.setDryWetMix(dryWet.get());

        return {std::move(dryWet), std::move(handles_)};
    }

    // No dry/wet wrapper - pipeline is the effects pipeline
    handles_.setEffectsPipeline(pipeline_.get());
    return {std::move(pipeline_), std::move(handles_)};
}

} // namespace dsp_core::audio_pipeline
