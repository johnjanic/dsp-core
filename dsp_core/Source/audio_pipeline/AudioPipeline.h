#pragma once

#include "AudioProcessingStage.h"
#include <vector>
#include <memory>

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
class AudioPipeline {
public:
    AudioPipeline() = default;

    /**
     * Add stage to end of pipeline.
     * MUST be called before prepareToPlay().
     */
    void addStage(std::unique_ptr<AudioProcessingStage> stage);

    /**
     * Prepare all stages for playback.
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock);

    /**
     * Process buffer through all stages in order.
     */
    void process(juce::AudioBuffer<double>& buffer);

    /**
     * Reset all stages.
     */
    void reset();

    /**
     * Get total latency of all stages combined.
     */
    int getTotalLatencySamples() const;

    /**
     * Get number of stages in pipeline.
     */
    int getNumStages() const { return static_cast<int>(stages_.size()); }

    /**
     * Clear all stages (for reconstruction).
     */
    void clear() { stages_.clear(); }

private:
    std::vector<std::unique_ptr<AudioProcessingStage>> stages_;
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
};

} // namespace dsp_core::audio_pipeline
