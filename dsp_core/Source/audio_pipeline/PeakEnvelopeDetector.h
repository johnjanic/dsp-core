#pragma once

#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Peak envelope detector with instant attack and exponential decay.
 *
 * Used for silence detection in dynamic DC compensation.
 * - Attack: Instant (zero latency for transient detection)
 * - Release: Exponential decay (smooth, musical envelope tracking)
 *
 * Thread Safety: Audio thread only (no synchronization needed)
 * CPU Cost: ~4 FLOPs per sample (abs, compare, multiply)
 */
struct PeakEnvelopeDetector {
    double peakLevel = 0.0;
    double decayCoeff = 0.0;

    /**
     * Configure decay time constant.
     *
     * @param sampleRate Current sample rate (may be oversampled)
     * @param decayTimeMs Time constant for exponential decay (default: 20ms)
     */
    void configure(double sampleRate, double decayTimeMs = 20.0) {
        // Exponential decay coefficient: e^(-1 / (sampleRate * timeInSeconds))
        // After decayTimeMs, envelope will be ~37% of original (1/e)
        decayCoeff = std::exp(-1.0 / (sampleRate * decayTimeMs / 1000.0));
    }

    /**
     * Process single sample.
     * Updates peak envelope with instant attack, exponential decay.
     *
     * @param inputSample Audio sample to process
     */
    void process(double inputSample) {
        double absInput = std::abs(inputSample);

        // Instant attack (max): zero latency for transients
        // Exponential decay: smooth release for musical envelope
        if (absInput > peakLevel) {
            peakLevel = absInput; // Attack
        } else {
            peakLevel *= decayCoeff; // Decay
        }
    }

    /**
     * Check if signal is near silence.
     *
     * Threshold: -54 dBFS (0.002 linear)
     * - Faster detection: envelope crosses threshold sooner
     * - Still well below noise floor of typical audio
     * - Reduces lag between audio stopping and compensation engaging
     *
     * @return true if peak level is below silence threshold
     */
    bool isNearSilence() const {
        return peakLevel < 0.002; // -54 dBFS
    }

    /**
     * Get current peak level (for debugging/visualization).
     *
     * @return Current envelope peak level [0.0, 1.0+]
     */
    double getPeakLevel() const {
        return peakLevel;
    }

    /**
     * Reset detector state (clear envelope).
     */
    void reset() {
        peakLevel = 0.0;
    }
};

} // namespace dsp_core::audio_pipeline
