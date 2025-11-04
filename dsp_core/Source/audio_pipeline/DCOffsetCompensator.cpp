#include "DCOffsetCompensator.h"
#include "../Services/ZeroCrossingSolver.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

DCOffsetCompensator::DCOffsetCompensator(LayeredTransferFunction& ltf)
    : ltf_(ltf)
{
    lastBiasUpdate_ = std::chrono::steady_clock::now();
}

void DCOffsetCompensator::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    sampleRate_ = sampleRate;

    // Configure fade controller with proper attack/release times
    fade_.configure(sampleRate);

    // Resize and configure per-channel envelopes (assume stereo by default)
    const int numChannels = 2;  // Will be resized in process() if needed
    channelEnvelopes_.resize(numChannels);

    for (auto& envelope : channelEnvelopes_) {
        envelope.configure(sampleRate, 100.0);  // 100ms decay time
    }

    // Compute initial bias (non-interactive)
    notifyTransferFunctionChanged(false);
}

void DCOffsetCompensator::process(juce::AudioBuffer<double>& buffer)
{
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Resize envelopes if channel count changed
    if (channelEnvelopes_.size() != static_cast<size_t>(numChannels)) {
        channelEnvelopes_.resize(numChannels);
        for (auto& envelope : channelEnvelopes_) {
            envelope.configure(sampleRate_, 100.0);
        }
    }

    // Early return if disabled: just apply transfer function without bias
    if (!enabled_.load(std::memory_order_acquire)) {
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                data[i] = ltf_.applyTransferFunction(data[i]);
            }
        }
        return;
    }

    // Load bias once per buffer (optimize atomic reads)
    const double bias = currentBias_.load(std::memory_order_acquire);

    // Process each channel
    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        auto& envelope = channelEnvelopes_[ch];

        for (int i = 0; i < numSamples; ++i) {
            // Update envelope detector
            envelope.process(data[i]);

            // Check if channel is near silence
            bool nearSilence = envelope.isNearSilence();

            // Update fade controller (updates target based on silence state)
            fade_.process(nearSilence);

            // Get current fade amount [0,1]
            double fadeAmount = fade_.getNextValue();

            // Apply bias: biasedInput = input + (fadeAmount * bias)
            double biasedInput = data[i] + (fadeAmount * bias);

            // NO CLAMPING - transparency over safety
            // Apply transfer function
            data[i] = ltf_.applyTransferFunction(biasedInput);
        }
    }
}

void DCOffsetCompensator::reset()
{
    // Reset all envelope detectors
    for (auto& envelope : channelEnvelopes_) {
        envelope.peakLevel = 0.0;
    }

    // Reset fade controller to zero (immediate)
    fade_.fadeAmount.setCurrentAndTargetValue(0.0);
}

void DCOffsetCompensator::notifyTransferFunctionChanged(bool isInteractiveEdit)
{
    // Rate limiting check
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastBiasUpdate_).count();

    const int debounceThreshold = isInteractiveEdit ? kDebounceInteractiveMs : kDebounceAutomationMs;

    if (elapsed < debounceThreshold) {
        // Too soon, skip update
        return;
    }

    lastBiasUpdate_ = now;

    // Solve for zero-crossing
    auto result = Services::ZeroCrossingSolver::solve(ltf_);

    // Store result with atomic release
    currentBias_.store(result.inputValue, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
