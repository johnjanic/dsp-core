#pragma once

#include "AudioProcessingStage.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace dsp_core::audio_pipeline {

/**
 * Serial pipeline of audio processing stages.
 *
 * Stages are executed in the order they were added.
 * Example:
 *   pipeline.addStage(makeGainStage());
 *   pipeline.addStage(makeWaveshaperStage());
 *   pipeline.addStage(makeHighpassStage());
 *   pipeline.process(buffer);  // Gain → Waveshaper → Highpass
 *
 * Thread safety:
 * - addStage() must be called before prepareToPlay() (setup phase only)
 * - process() is thread-safe after setup
 */
class AudioPipeline : public AudioProcessingStage {
public:
    AudioPipeline() = default;

    /**
     * Add stage to end of pipeline with optional tag.
     * MUST be called before prepareToPlay().
     * @param stage Stage to add
     * @param tag Optional tag for retrieval (auto-generated if empty)
     */
    void addStage(std::unique_ptr<AudioProcessingStage> stage, const std::string& tag = "");

    /**
     * Prepare all stages for playback.
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;

    /**
     * Process buffer through all stages in order.
     */
    void process(juce::AudioBuffer<double>& buffer) override;

    /**
     * Reset all stages.
     */
    void reset() override;

    /**
     * Get name of pipeline (lists all stages).
     */
    juce::String getName() const override;

    /**
     * Get total latency of all stages combined.
     */
    int getLatencySamples() const override;

    /**
     * Get total latency of all stages combined (legacy name).
     */
    int getTotalLatencySamples() const { return getLatencySamples(); }

    /**
     * Get number of stages in pipeline.
     */
    int getNumStages() const { return static_cast<int>(stages_.size()); }

    /**
     * Clear all stages (for reconstruction).
     */
    void clear();

    /**
     * Get stage by tag (type-erased).
     * @param tag Tag used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found
     */
    AudioProcessingStage* getStage(const std::string& tag);

    /**
     * Get stage by tag (typed).
     * @tparam StageType Expected stage type
     * @param tag Tag used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found or type mismatch
     */
    template<typename StageType>
    StageType* getStage(const std::string& tag) {
        auto* stage = getStage(tag);
        return dynamic_cast<StageType*>(stage);
    }

private:
    std::vector<std::unique_ptr<AudioProcessingStage>> stages_;
    std::unordered_map<std::string, size_t> tagToIndex_;  // Tag -> stages_ index
    int autoTagCounter_ = 0;
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
};

} // namespace dsp_core::audio_pipeline
