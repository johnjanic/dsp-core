#pragma once

#include "AudioProcessingStage.h"
#include "../LayeredTransferFunction.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>
#include <chrono>

namespace dsp_core::audio_pipeline {

/**
 * Adaptive DC offset compensation for transfer functions where f(0) ≠ 0.
 *
 * Problem: When silent input (x=0) produces non-zero output, audio contains DC offset.
 * Solution: Apply input bias during silence to shift operating point toward f(x*)≈0.
 *
 * Algorithm:
 *   1. Detect near-silence per channel (peak envelope < -60 dBFS)
 *   2. Fade in bias over 200ms (prevents pumping during natural decay)
 *   3. Apply: output = f(input + N*bias), where N ∈ [0,1]
 *   4. Fade out bias in 10ms when signal returns (preserves transients)
 *
 * Thread Safety:
 *   - currentBias_: atomic (UI writes, audio reads)
 *   - Envelope state: per-instance (audio thread only)
 *   - ZeroCrossingSolver: UI thread only (called from notifyTransferFunctionChanged)
 *
 * CPU Cost: ~1-2% overhead (envelope detection + bias addition)
 */
class DCOffsetCompensator : public AudioProcessingStage {
public:
    /**
     * Peak envelope detector with instant attack and exponential decay.
     * Used for silence detection (threshold: -60 dBFS).
     */
    struct PeakEnvelopeDetector {
        double peakLevel = 0.0;
        double decayCoeff = 0.0;

        void configure(double sampleRate, double decayTimeMs) {
            // Exponential decay: coeff = exp(-1 / (sampleRate * timeInSeconds))
            decayCoeff = std::exp(-1.0 / (sampleRate * decayTimeMs / 1000.0));
        }

        void process(double inputSample) {
            double absInput = std::abs(inputSample);
            // Instant attack (max), exponential decay
            peakLevel = std::max(absInput, peakLevel * decayCoeff);
        }

        bool isNearSilence() const {
            return peakLevel < 0.001;  // -60 dBFS
        }

        double getPeakLevel() const {
            return peakLevel;
        }
    };

    /**
     * Bias fade controller: smooth transitions between compensated/uncompensated states.
     * Attack: 200ms (slow fade-in prevents pumping during decay)
     * Release: 10ms (fast fade-out preserves transient response)
     */
    struct BiasFadeController {
        juce::LinearSmoothedValue<double> fadeAmount;

        void configure(double sampleRate) {
            fadeAmount.reset(sampleRate, 0.010);  // 10ms ramp for reset
        }

        void process(bool isNearSilence) {
            double targetValue = isNearSilence ? 1.0 : 0.0;

            // Set ramp time based on direction
            if (isNearSilence && fadeAmount.getTargetValue() < 0.5) {
                // Transitioning to silence: 200ms attack
                fadeAmount.reset(fadeAmount.getCurrentValue());
                fadeAmount.setTargetValue(targetValue);
                fadeAmount.reset(fadeAmount.getCurrentValue());
                fadeAmount.setTargetValue(targetValue);
                // Note: setTargetValue uses the ramp time set in configure()
                // We'll handle the attack time in the actual implementation
            } else if (!isNearSilence && fadeAmount.getTargetValue() > 0.5) {
                // Transitioning to signal: 10ms release
                fadeAmount.reset(fadeAmount.getCurrentValue());
                fadeAmount.setTargetValue(targetValue);
            } else {
                fadeAmount.setTargetValue(targetValue);
            }
        }

        double getNextValue() {
            return fadeAmount.getNextValue();
        }

        double getCurrentValue() const {
            return fadeAmount.getCurrentValue();
        }
    };

    /**
     * Constructor: takes reference to transfer function.
     * Transfer function must outlive this stage (typically both owned by processor).
     */
    explicit DCOffsetCompensator(LayeredTransferFunction& ltf);

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "DCOffsetCompensator"; }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) { enabled_.store(shouldBeEnabled, std::memory_order_release); }
    bool isEnabled() const { return enabled_.load(std::memory_order_acquire); }

    /**
     * Notify that transfer function has changed.
     * Recomputes optimal bias using ZeroCrossingSolver.
     *
     * Thread Safety: UI thread only (solver is not real-time safe)
     * Rate Limiting: 16ms for interactive edits, 50ms for automation
     *
     * @param isInteractiveEdit True for paint strokes (faster update), false for presets/automation
     */
    void notifyTransferFunctionChanged(bool isInteractiveEdit);

    /**
     * Get current bias value (for UI display/debugging).
     * Thread-safe: uses atomic load.
     */
    double getCurrentBias() const { return currentBias_.load(std::memory_order_acquire); }

    /**
     * Check if compensation is currently active (faded in).
     * Used for UI indicators (e.g., show "COMP" light when active).
     */
    bool isCompensating() const { return fade_.getCurrentValue() > 0.01; }

private:
    LayeredTransferFunction& ltf_;
    std::atomic<bool> enabled_{true};  // Default: ON (safety feature)
    std::atomic<double> currentBias_{0.0};  // UI writes, audio reads

    // Per-channel state (audio thread only, no synchronization needed)
    std::vector<PeakEnvelopeDetector> channelEnvelopes_;
    BiasFadeController fade_;
    double sampleRate_ = 48000.0;

    // Rate limiting for bias updates (UI thread only)
    std::chrono::steady_clock::time_point lastBiasUpdate_;
    static constexpr int kDebounceInteractiveMs = 16;   // ~60 Hz for paint strokes
    static constexpr int kDebounceAutomationMs = 50;    // ~20 Hz for automation
};

} // namespace dsp_core::audio_pipeline
