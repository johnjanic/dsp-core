#pragma once

#include "AudioProcessingStage.h"
#include "SilenceDetector.h"
#include "BiasFadeController.h"
#include "../LayeredTransferFunction.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <chrono>

namespace dsp_core::audio_pipeline {

/**
 * Dynamic output biasing for adaptive DC offset compensation.
 *
 * Problem: When transfer function has f(0) ≠ 0, silent input produces DC offset.
 * Solution: Subtract DC offset adaptively during silence: output = y - N × B
 *   - y = f(x) (waveshaper output)
 *   - B = f(0) (cached DC offset, updated on transfer function change)
 *   - N ∈ [0,1] (fade amount from BiasFadeController)
 *
 * Algorithm:
 *   1. SilenceDetector (separate stage) detects input silence
 *   2. BiasFadeController smoothly fades bias in/out (300ms attack, 10ms release)
 *   3. Apply: output -= N × B (simple subtraction)
 *
 * Pipeline Position: After WaveshapingStage (inside oversampling wrapper)
 *
 * Thread Safety:
 *   - Audio thread: Reads cachedBias_ (atomic), reads SilenceDetector flag
 *   - UI thread: Writes cachedBias_ via notifyTransferFunctionChanged()
 *   - No locks, lock-free design
 *
 * CPU Cost: ~3 FLOPs per sample (fade update + bias subtraction)
 */
class DynamicOutputBiasing : public AudioProcessingStage {
public:
    /**
     * Constructor with dependency injection.
     *
     * @param ltf Reference to transfer function (must outlive this stage)
     * @param silenceDetector Reference to silence detector (must run before this stage)
     */
    explicit DynamicOutputBiasing(
        LayeredTransferFunction& ltf,
        SilenceDetector& silenceDetector
    );

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "DynamicOutputBiasing"; }

    /**
     * Enable/disable compensation (for A/B testing).
     *
     * Thread-safe: uses atomic storage.
     *
     * @param shouldBeEnabled true to enable compensation
     */
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    /**
     * Check if compensation is enabled.
     *
     * @return true if enabled
     */
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Notify that transfer function has changed.
     * Recomputes cached bias: B = f(0)
     *
     * Thread Safety: UI thread only (not real-time safe due to function evaluation)
     * Rate Limiting: Max 20 updates/sec (50ms debounce)
     */
    void notifyTransferFunctionChanged();

    /**
     * Get current cached bias value (for UI display/debugging).
     *
     * Thread-safe: uses atomic load.
     *
     * @return Current DC offset bias value
     */
    double getCurrentBias() const {
        return cachedBias_.load(std::memory_order_acquire);
    }

    /**
     * Get current fade amount (for UI display/debugging).
     *
     * @return Current fade value [0.0, 1.0]
     */
    double getCurrentFade() const {
        return fade_.getCurrentValue();
    }

    /**
     * Check if compensation is currently active (faded in).
     *
     * @return true if fade amount > 1% (compensation active)
     */
    bool isCompensating() const {
        return fade_.getCurrentValue() > 0.01;
    }

private:
    /**
     * Update cached bias value (called from notifyTransferFunctionChanged).
     * UI thread only.
     */
    void updateBias();

    LayeredTransferFunction& ltf_;
    SilenceDetector& silenceDetector_;

    // Algorithm components
    BiasFadeController fade_;

    // State
    double sampleRate_ = 44100.0;
    std::atomic<bool> enabled_{true};  // Default: ON (safety feature)
    std::atomic<double> cachedBias_{0.0};  // UI writes, audio reads

    // Rate limiting for bias updates (UI thread only)
    std::chrono::steady_clock::time_point lastBiasUpdate_;
    static constexpr int kDebounceMs = 50;  // ~20 Hz max update rate
};

} // namespace dsp_core::audio_pipeline
