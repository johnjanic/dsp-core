#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <thread>

using namespace dsp_core::audio_pipeline;

// ============================================================================
// PeakEnvelopeDetector Tests
// ============================================================================

class PeakEnvelopeDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector_.configure(48000.0, 100.0);  // 100ms decay at 48kHz
    }

    PeakEnvelopeDetector detector_;
};

TEST_F(PeakEnvelopeDetectorTest, InstantAttack) {
    // Process zero, then sudden peak
    detector_.process(0.0);
    EXPECT_DOUBLE_EQ(detector_.getPeakLevel(), 0.0);

    detector_.process(0.5);
    EXPECT_DOUBLE_EQ(detector_.getPeakLevel(), 0.5);  // Instant attack

    detector_.process(1.0);
    EXPECT_DOUBLE_EQ(detector_.getPeakLevel(), 1.0);  // Higher peak
}

TEST_F(PeakEnvelopeDetectorTest, ExponentialDecay) {
    // Set peak
    detector_.process(1.0);
    EXPECT_DOUBLE_EQ(detector_.getPeakLevel(), 1.0);

    // Process silence, should decay exponentially
    double previousLevel = detector_.getPeakLevel();
    for (int i = 0; i < 100; ++i) {
        detector_.process(0.0);
        double currentLevel = detector_.getPeakLevel();
        EXPECT_LT(currentLevel, previousLevel);  // Should decay
        previousLevel = currentLevel;
    }

    // After many samples (10ms at 48kHz = 480 samples), should decay significantly
    for (int i = 0; i < 5000; ++i) {
        detector_.process(0.0);
    }
    EXPECT_LT(detector_.getPeakLevel(), 0.5);
}

TEST_F(PeakEnvelopeDetectorTest, SilenceThreshold) {
    // Below threshold
    detector_.process(0.0005);  // -66 dBFS
    for (int i = 0; i < 1000; ++i) {
        detector_.process(0.0);
    }
    EXPECT_TRUE(detector_.isNearSilence());

    // Above threshold
    detector_.process(0.005);  // -46 dBFS
    EXPECT_FALSE(detector_.isNearSilence());
}

TEST_F(PeakEnvelopeDetectorTest, Reset) {
    detector_.process(1.0);
    EXPECT_GT(detector_.getPeakLevel(), 0.0);

    detector_.reset();
    EXPECT_DOUBLE_EQ(detector_.getPeakLevel(), 0.0);
}

// ============================================================================
// BiasFadeController Tests
// ============================================================================

class BiasFadeControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        controller_.configure(48000.0);
    }

    BiasFadeController controller_;
};

TEST_F(BiasFadeControllerTest, InitialState) {
    EXPECT_DOUBLE_EQ(controller_.getCurrentValue(), 0.0);
}

TEST_F(BiasFadeControllerTest, SlowAttackToSilence) {
    // Transition to silence (should use 300ms attack)
    controller_.process(true);  // isNearSilence = true

    // After a few samples, should be ramping up slowly
    double previousValue = controller_.getCurrentValue();
    for (int i = 0; i < 1000; ++i) {
        controller_.process(true);
        double currentValue = controller_.getNextValue();
        EXPECT_GE(currentValue, previousValue);  // Should increase
        previousValue = currentValue;
    }

    // After 300ms (14400 samples at 48kHz), should be close to 1.0
    for (int i = 0; i < 14400; ++i) {
        controller_.process(true);
        controller_.getNextValue();
    }
    EXPECT_GT(controller_.getCurrentValue(), 0.95);
}

TEST_F(BiasFadeControllerTest, FastReleaseFromSilence) {
    // First, fade in to silence
    for (int i = 0; i < 20000; ++i) {
        controller_.process(true);
        controller_.getNextValue();
    }
    EXPECT_GT(controller_.getCurrentValue(), 0.95);

    // Now transition to signal (should use 10ms release)
    controller_.process(false);  // isNearSilence = false

    // After 10ms (480 samples at 48kHz), should be close to 0.0
    for (int i = 0; i < 480; ++i) {
        controller_.process(false);
        controller_.getNextValue();
    }
    EXPECT_LT(controller_.getCurrentValue(), 0.1);
}

TEST_F(BiasFadeControllerTest, Reset) {
    // Fade in
    for (int i = 0; i < 20000; ++i) {
        controller_.process(true);
        controller_.getNextValue();
    }
    EXPECT_GT(controller_.getCurrentValue(), 0.9);

    controller_.reset();
    EXPECT_DOUBLE_EQ(controller_.getCurrentValue(), 0.0);
}

// ============================================================================
// SilenceDetector Tests
// ============================================================================

class SilenceDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        detector_ = std::make_unique<SilenceDetector>();
        detector_->prepareToPlay(48000.0, 512);
    }

    std::unique_ptr<SilenceDetector> detector_;
};

TEST_F(SilenceDetectorTest, DetectsSilence) {
    juce::AudioBuffer<double> buffer(2, 48000);  // 1 second of silence
    buffer.clear();

    detector_->process(buffer);

    // Should detect silence after processing
    EXPECT_TRUE(detector_->isNearSilence());
}

TEST_F(SilenceDetectorTest, DetectsSignal) {
    juce::AudioBuffer<double> buffer(2, 512);
    buffer.clear();

    // Add signal to both channels
    buffer.setSample(0, 0, 0.5);
    buffer.setSample(1, 0, 0.5);

    detector_->process(buffer);

    // Should NOT detect silence
    EXPECT_FALSE(detector_->isNearSilence());
}

TEST_F(SilenceDetectorTest, AllChannelsMustBeSilent) {
    juce::AudioBuffer<double> buffer(2, 512);
    buffer.clear();

    // Only one channel has signal
    buffer.setSample(0, 0, 0.5);  // Left: signal
    // Right: silence

    detector_->process(buffer);

    // Should NOT detect silence (requires ALL channels silent)
    EXPECT_FALSE(detector_->isNearSilence());
}

TEST_F(SilenceDetectorTest, PassThrough) {
    juce::AudioBuffer<double> buffer(2, 512);

    // Fill with test pattern
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(ch, i, std::sin(2.0 * M_PI * 1000.0 * i / 48000.0));
        }
    }

    // Store original values
    std::vector<double> originalValues;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            originalValues.push_back(buffer.getSample(ch, i));
        }
    }

    // Process
    detector_->process(buffer);

    // Verify buffer unchanged (pass-through)
    int idx = 0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, i), originalValues[idx++]);
        }
    }
}

TEST_F(SilenceDetectorTest, Reset) {
    // Process signal
    juce::AudioBuffer<double> buffer(2, 512);
    buffer.setSample(0, 0, 1.0);
    detector_->process(buffer);
    EXPECT_FALSE(detector_->isNearSilence());

    // Reset
    detector_->reset();

    // Process silence
    buffer.clear();
    detector_->process(buffer);
    EXPECT_TRUE(detector_->isNearSilence());
}

// ============================================================================
// DynamicOutputBiasing Tests
// ============================================================================

class DynamicOutputBiasingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ltf_ = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
        silenceDetector_ = std::make_unique<SilenceDetector>();
        biasing_ = std::make_unique<DynamicOutputBiasing>(*ltf_, *silenceDetector_);

        silenceDetector_->prepareToPlay(48000.0, 512);
        // Note: Don't call biasing_->prepareToPlay() here - let tests set up transfer function first
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf_;
    std::unique_ptr<SilenceDetector> silenceDetector_;
    std::unique_ptr<DynamicOutputBiasing> biasing_;
};

TEST_F(DynamicOutputBiasingTest, ComputesBiasCorrectly) {
    // Set up transfer function with DC offset
    // Create asymmetric function that has f(0) ≠ 0
    for (int i = 0; i < 256; ++i) {
        double x = ltf_->normalizeIndex(i);  // Map to [-1, 1]
        // Asymmetric tanh-like function
        ltf_->setBaseLayerValue(i, std::tanh(x + 0.5));
    }
    ltf_->updateComposite();

    // Prepare after setting up transfer function
    biasing_->prepareToPlay(48000.0, 512);

    // Verify bias is non-zero (DC offset exists)
    double bias = biasing_->getCurrentBias();
    double expected_f0 = ltf_->applyTransferFunction(0.0);
    EXPECT_NEAR(bias, expected_f0, 0.01);
}

TEST_F(DynamicOutputBiasingTest, CompensatesDuringSlowFade) {
    // Set up transfer function with DC offset
    for (int i = 0; i < 256; ++i) {
        double x = ltf_->normalizeIndex(i);
        ltf_->setBaseLayerValue(i, std::tanh(x + 0.5));  // Asymmetric function
    }
    ltf_->updateComposite();

    // Prepare after setting up transfer function
    biasing_->prepareToPlay(48000.0, 512);

    // Get the bias value (what f(0) evaluates to)
    double bias = biasing_->getCurrentBias();

    // Process silence through detector
    juce::AudioBuffer<double> silenceBuffer(2, 48000);
    silenceBuffer.clear();
    silenceDetector_->process(silenceBuffer);

    // Create output buffer with constant DC offset matching our bias
    juce::AudioBuffer<double> outputBuffer(2, 48000);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 48000; ++i) {
            outputBuffer.setSample(ch, i, bias);  // DC offset from waveshaper
        }
    }

    // Process through biasing
    biasing_->process(outputBuffer);

    // After fade-in (300ms = 14400 samples), output should be close to zero
    double finalMean = 0.0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 40000; i < 48000; ++i) {  // Check last 8000 samples
            finalMean += outputBuffer.getSample(ch, i);
        }
    }
    finalMean /= (2 * 8000);

    EXPECT_NEAR(finalMean, 0.0, 0.05);  // Should compensate most of the DC
}

TEST_F(DynamicOutputBiasingTest, TransientDisablesCompensation) {
    // Set up transfer function
    for (int i = 0; i < 256; ++i) {
        ltf_->setBaseLayerValue(i, 0.3);
    }
    ltf_->updateComposite();
    biasing_->prepareToPlay(48000.0, 512);

    // First: silence (fade should ramp up)
    juce::AudioBuffer<double> silenceBuffer(2, 24000);  // 0.5s
    silenceBuffer.clear();
    silenceDetector_->process(silenceBuffer);

    juce::AudioBuffer<double> outputBuffer(2, 24000);
    for (int i = 0; i < 24000; ++i) {
        outputBuffer.setSample(0, i, 0.3);
        outputBuffer.setSample(1, i, 0.3);
    }
    biasing_->process(outputBuffer);

    // Verify fade is active
    EXPECT_GT(biasing_->getCurrentFade(), 0.3);

    // Then: sharp transient
    juce::AudioBuffer<double> transientInputBuffer(2, 512);
    transientInputBuffer.clear();
    transientInputBuffer.setSample(0, 0, 1.0);  // Sharp transient
    transientInputBuffer.setSample(1, 0, 1.0);
    silenceDetector_->process(transientInputBuffer);

    juce::AudioBuffer<double> transientOutputBuffer(2, 512);
    for (int i = 0; i < 512; ++i) {
        transientOutputBuffer.setSample(0, i, 0.3);
        transientOutputBuffer.setSample(1, i, 0.3);
    }
    biasing_->process(transientOutputBuffer);

    // Verify fade drops quickly (within 10ms = 480 samples)
    for (int i = 0; i < 10; ++i) {
        juce::AudioBuffer<double> buf(2, 512);
        buf.clear();
        silenceDetector_->process(buf);
        biasing_->process(buf);
    }

    EXPECT_LT(biasing_->getCurrentFade(), 0.2);  // Should be near 0
}

TEST_F(DynamicOutputBiasingTest, CanBeDisabled) {
    // Set up DC offset
    for (int i = 0; i < 256; ++i) {
        ltf_->setBaseLayerValue(i, 0.5);
    }
    ltf_->updateComposite();
    biasing_->prepareToPlay(48000.0, 512);

    // Disable
    biasing_->setEnabled(false);
    EXPECT_FALSE(biasing_->isEnabled());

    // Process with silence
    juce::AudioBuffer<double> silenceBuffer(2, 512);
    silenceBuffer.clear();
    silenceDetector_->process(silenceBuffer);

    juce::AudioBuffer<double> outputBuffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            outputBuffer.setSample(ch, i, 0.5);
        }
    }

    biasing_->process(outputBuffer);

    // Should NOT compensate (bypass)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_DOUBLE_EQ(outputBuffer.getSample(ch, i), 0.5);
        }
    }
}

TEST_F(DynamicOutputBiasingTest, RateLimiting) {
    // Set initial transfer function - constant 0.2
    for (int i = 0; i < 256; ++i) {
        ltf_->setBaseLayerValue(i, 0.2);
    }
    ltf_->updateComposite();

    // Prepare and get initial bias
    biasing_->prepareToPlay(48000.0, 512);
    double bias1 = biasing_->getCurrentBias();
    // Constant 0.2 normalizes to 1.0, so f(0) = 1.0
    EXPECT_NEAR(std::abs(bias1), 1.0, 0.01);

    // Immediate second call (should be debounced, bias should not change)
    for (int i = 0; i < 256; ++i) {
        ltf_->setBaseLayerValue(i, -0.5);  // Change to -0.5
    }
    ltf_->updateComposite();
    biasing_->notifyTransferFunctionChanged();
    double bias2 = biasing_->getCurrentBias();

    // Should be same value (rate limited)
    EXPECT_NEAR(bias2, bias1, 0.01);

    // Wait 60ms
    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Now should update
    for (int i = 0; i < 256; ++i) {
        ltf_->setBaseLayerValue(i, -0.8);  // Change to -0.8
    }
    ltf_->updateComposite();

    biasing_->notifyTransferFunctionChanged();
    double bias3 = biasing_->getCurrentBias();

    // Should have updated (should be -1.0 now, constant negative normalizes to -1.0)
    EXPECT_NEAR(std::abs(bias3), 1.0, 0.01);
    EXPECT_NE(bias1, bias3);  // Should have opposite sign
}
