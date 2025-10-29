#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Bias fade controller with adaptive attack/release timing.
 *
 * Smoothly transitions bias amount between 0 (signal active) and 1 (silence detected).
 * Uses exponential smoothing for natural-sounding envelope.
 *
 * Attack/Release Philosophy:
 * - Fast attack (50ms): Bias fades in quickly during silence
 * - Very fast release (2ms): Near-instant removal on transients, preserves attack transients
 *
 * Thread Safety: Audio thread only (no synchronization needed)
 * CPU Cost: ~3 FLOPs per sample (exponential smoothing)
 */
struct BiasFadeController {
    double currentValue_ = 0.0;
    double targetValue_ = 0.0;
    double attackCoeff_ = 0.0;
    double releaseCoeff_ = 0.0;
    double sampleRate_ = 44100.0;

    // Timing constants (conservative, tunable based on audio tests)
    static constexpr double attackTimeSeconds = 0.05;   // 50ms - very fast fade-in during silence
    static constexpr double releaseTimeSeconds = 0.002; // 2ms - near-instant transient response

    /**
     * Configure controller for given sample rate.
     *
     * Calculates exponential coefficients for attack/release times.
     * Uses formula: coeff = exp(-1 / (time * sampleRate))
     *
     * @param sampleRate Current sample rate (may be oversampled)
     */
    void configure(double sampleRate) {
        sampleRate_ = sampleRate;

        // Calculate exponential smoothing coefficients
        // Target: reach 99% of target value in specified time
        // Formula: coeff = exp(-ln(100) / (time * sampleRate))
        attackCoeff_ = std::exp(-4.605 / (attackTimeSeconds * sampleRate));  // ln(100) ≈ 4.605
        releaseCoeff_ = std::exp(-4.605 / (releaseTimeSeconds * sampleRate));
    }

    /**
     * Update fade state based on silence detection.
     *
     * Uses exponential smoothing with different coefficients for attack/release:
     * - Entering silence: attack coefficient (slower)
     * - Exiting silence: release coefficient (faster)
     *
     * Formula: current = current * coeff + target * (1 - coeff)
     *
     * @param isNearSilence true if signal is near silence threshold
     */
    void process(bool isNearSilence) {
        // Set target value
        targetValue_ = isNearSilence ? 1.0 : 0.0;

        // Select coefficient based on whether we're approaching or leaving target
        double coeff;
        if (currentValue_ < targetValue_) {
            // Approaching silence (fade in) - use attack
            coeff = attackCoeff_;
        } else {
            // Leaving silence (fade out) - use release
            coeff = releaseCoeff_;
        }

        // Exponential smoothing: smooth towards target
        currentValue_ = currentValue_ * coeff + targetValue_ * (1.0 - coeff);
    }

    /**
     * Get next fade value (call once per sample).
     *
     * @return Current fade amount [0.0, 1.0]
     */
    double getNextValue() {
        return currentValue_;
    }

    /**
     * Get current fade value without advancing (for monitoring).
     *
     * @return Current fade amount [0.0, 1.0]
     */
    double getCurrentValue() const {
        return currentValue_;
    }

    /**
     * Reset fade state to zero (no compensation).
     */
    void reset() {
        currentValue_ = 0.0;
        targetValue_ = 0.0;
    }
};

} // namespace dsp_core::audio_pipeline
