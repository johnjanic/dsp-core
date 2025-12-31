#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/InputPeakTracker.h"
#include "../dsp_core/Source/audio_pipeline/AudioPipeline.h"
#include "../dsp_core/Source/audio_pipeline/GainStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>

using namespace dsp_core::audio_pipeline;

class InputPeakTrackerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        peakStorage_.store(0.0, std::memory_order_release);
        tracker_ = std::make_unique<InputPeakTracker>(peakStorage_);
        tracker_->prepareToPlay(44100.0, 512);
    }

    std::atomic<double> peakStorage_{0.0};
    std::unique_ptr<InputPeakTracker> tracker_;
};

TEST_F(InputPeakTrackerTest, Silence_ProducesZeroPeak) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear(); // All zeros

    tracker_->process(buffer);

    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.0);
}

TEST_F(InputPeakTrackerTest, PositiveSignal_TrackedCorrectly) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();

    // Set peak value to 0.75 in middle of buffer
    buffer.setSample(0, 32, 0.75);
    buffer.setSample(1, 32, 0.5); // Different value in other channel

    tracker_->process(buffer);

    // Should capture maximum across all channels
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.75);
}

TEST_F(InputPeakTrackerTest, NegativeSignal_AbsoluteValueTracked) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();

    // Set negative peak value
    buffer.setSample(0, 32, -0.85);
    buffer.setSample(1, 32, -0.3);

    tracker_->process(buffer);

    // Should capture absolute value (0.85, not -0.85)
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.85);
}

TEST_F(InputPeakTrackerTest, MixedPolarity_MaxAbsoluteValueTracked) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();

    // Mix of positive and negative values
    buffer.setSample(0, 10, 0.6);  // Positive
    buffer.setSample(0, 20, -0.9); // Negative (larger abs value)
    buffer.setSample(1, 30, 0.4);  // Positive (smaller)

    tracker_->process(buffer);

    // Should capture max absolute value (0.9)
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.9);
}

TEST_F(InputPeakTrackerTest, ClippedSignal_CapturesFullScale) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();

    // Clipped signal at +1.0 (full scale)
    for (int i = 0; i < 10; ++i) {
        buffer.setSample(0, i, 1.0);
        buffer.setSample(1, i, 1.0);
    }

    tracker_->process(buffer);

    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 1.0);
}

TEST_F(InputPeakTrackerTest, ExceedingFullScale_CapturesOverload) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();

    // Signal exceeding full scale (e.g., from input gain boost)
    buffer.setSample(0, 32, 1.5);
    buffer.setSample(1, 32, 0.8);

    tracker_->process(buffer);

    // Should capture values > 1.0 (important for visualizing overload)
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 1.5);
}

TEST_F(InputPeakTrackerTest, MultipleChannels_TracksGlobalPeak) {
    juce::AudioBuffer<double> buffer(8, 64); // 8 channels
    buffer.clear();

    // Different peak in each channel
    buffer.setSample(0, 10, 0.1);
    buffer.setSample(1, 20, 0.2);
    buffer.setSample(2, 30, 0.3);
    buffer.setSample(3, 40, 0.95); // Maximum
    buffer.setSample(4, 50, 0.4);
    buffer.setSample(5, 0, 0.5);
    buffer.setSample(6, 15, 0.6);
    buffer.setSample(7, 25, 0.7);

    tracker_->process(buffer);

    // Should find maximum across all 8 channels
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.95);
}

TEST_F(InputPeakTrackerTest, SingleSample_HandledCorrectly) {
    juce::AudioBuffer<double> buffer(1, 1);
    buffer.setSample(0, 0, 0.42);

    tracker_->process(buffer);

    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.42);
}

TEST_F(InputPeakTrackerTest, LargeBuffer_TracksCorrectly) {
    juce::AudioBuffer<double> buffer(2, 2048); // Large buffer
    buffer.clear();

    // Generate sine wave with known peak
    const double freq = 1000.0;
    const double sampleRate = 44100.0;
    const double amplitude = 0.7;

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 2048; ++i) {
            buffer.setSample(ch, i, amplitude * std::sin(2.0 * M_PI * freq * i / sampleRate));
        }
    }

    tracker_->process(buffer);

    // Peak should be close to amplitude (within numerical precision)
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), amplitude, 0.01);
}

TEST_F(InputPeakTrackerTest, ConsecutiveBuffers_OverwritesPreviousPeak) {
    juce::AudioBuffer<double> buffer(2, 64);

    // First buffer with peak 0.8
    buffer.clear();
    buffer.setSample(0, 32, 0.8);
    tracker_->process(buffer);
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.8);

    // Second buffer with lower peak 0.3
    buffer.clear();
    buffer.setSample(0, 32, 0.3);
    tracker_->process(buffer);
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.3);

    // Third buffer with higher peak 0.95
    buffer.clear();
    buffer.setSample(0, 32, 0.95);
    tracker_->process(buffer);
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.95);
}

TEST_F(InputPeakTrackerTest, Reset_ClearsPeak) {
    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.9);

    tracker_->process(buffer);
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.9);

    tracker_->reset();
    EXPECT_DOUBLE_EQ(peakStorage_.load(std::memory_order_acquire), 0.0);
}

// =============================================================================
// Integration Tests (with Pipeline)
// =============================================================================

class InputPeakTrackerIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        peakStorage_.store(0.0, std::memory_order_release);

        // Create pipeline: InputGain -> InputPeakTracker
        // This mirrors the real plugin architecture
        pipeline_ = std::make_unique<AudioPipeline>();

        auto gainStage = std::make_unique<GainStage>();
        inputGainStage_ = gainStage.get();
        pipeline_->addStage(std::move(gainStage), StageTag::InputGain);

        auto tracker = std::make_unique<InputPeakTracker>(peakStorage_);
        pipeline_->addStage(std::move(tracker), StageTag::InputPeakTracker);

        pipeline_->prepareToPlay(44100.0, 512);
    }

    std::atomic<double> peakStorage_{0.0};
    std::unique_ptr<AudioPipeline> pipeline_;
    GainStage* inputGainStage_ = nullptr;
};

TEST_F(InputPeakTrackerIntegrationTest, TracksAfterInputGain_UnityGain) {
    inputGainStage_->setGainDB(0.0); // Unity gain (0 dB = 1.0x)

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.5);

    pipeline_->process(buffer);

    // With unity gain, peak should be unchanged
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), 0.5, 1e-6);
}

TEST_F(InputPeakTrackerIntegrationTest, TracksAfterInputGain_Boost) {
    inputGainStage_->setGainDB(6.0); // +6 dB ≈ 2x boost

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.5);

    pipeline_->process(buffer);

    // With +6 dB gain, peak should be ~1.0 (0.5 * 2.0)
    const double expectedPeak = 0.5 * std::pow(10.0, 6.0 / 20.0); // 0.5 * linear gain
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), expectedPeak, 0.01);
}

TEST_F(InputPeakTrackerIntegrationTest, TracksAfterInputGain_Cut) {
    inputGainStage_->setGainDB(-6.0); // -6 dB ≈ 0.5x cut

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.8);

    pipeline_->process(buffer);

    // With -6 dB gain, peak should be ~0.4 (0.8 * 0.5)
    const double expectedPeak = 0.8 * std::pow(10.0, -6.0 / 20.0);
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), expectedPeak, 0.01);
}

TEST_F(InputPeakTrackerIntegrationTest, TracksAfterInputGain_LargeBoost) {
    inputGainStage_->setGainDB(12.0); // +12 dB ≈ 4x boost

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.4);

    pipeline_->process(buffer);

    // With +12 dB gain, peak should be ~1.6 (0.4 * 4.0) - exceeding full scale
    const double expectedPeak = 0.4 * std::pow(10.0, 12.0 / 20.0);
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), expectedPeak, 0.01);
    EXPECT_GT(peakStorage_.load(std::memory_order_acquire), 1.0); // Verify overload captured
}

TEST_F(InputPeakTrackerIntegrationTest, TracksAfterInputGain_NearSilence) {
    inputGainStage_->setGainDB(-60.0); // -60 dB (near silence)

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.5);

    pipeline_->process(buffer);

    // With -60 dB gain, peak should be very small
    const double expectedPeak = 0.5 * std::pow(10.0, -60.0 / 20.0);
    EXPECT_NEAR(peakStorage_.load(std::memory_order_acquire), expectedPeak, 1e-6);
    EXPECT_LT(peakStorage_.load(std::memory_order_acquire), 0.001);
}

TEST_F(InputPeakTrackerIntegrationTest, RealWorldScenario_MicrophoneInput) {
    // Simulate microphone input scenario:
    // - Moderate input signal (0.3 amplitude)
    // - +12 dB input gain (common for quiet sources)
    // - Continuous audio over multiple buffers

    inputGainStage_->setGainDB(12.0);

    const double inputAmplitude = 0.3;
    const double expectedPeak = inputAmplitude * std::pow(10.0, 12.0 / 20.0);

    // Process multiple buffers to simulate continuous audio
    for (int block = 0; block < 10; ++block) {
        juce::AudioBuffer<double> buffer(2, 512);

        // Generate varying signal (sine wave)
        const double freq = 440.0;
        const double sampleRate = 44100.0;
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 512; ++i) {
                const int globalSample = block * 512 + i;
                buffer.setSample(ch, i, inputAmplitude * std::sin(2.0 * M_PI * freq * globalSample / sampleRate));
            }
        }

        pipeline_->process(buffer);

        // Peak should stabilize around expected value
        const double measuredPeak = peakStorage_.load(std::memory_order_acquire);
        EXPECT_NEAR(measuredPeak, expectedPeak, 0.1)
            << "Block " << block << " peak mismatch";
    }
}

TEST_F(InputPeakTrackerIntegrationTest, ThreadSafety_ConcurrentReads) {
    // This test verifies memory ordering is correct for concurrent audio thread writes
    // and UI thread reads (though we can't truly test concurrency in a single-threaded test)

    inputGainStage_->setGainDB(0.0);

    juce::AudioBuffer<double> buffer(2, 64);
    buffer.clear();
    buffer.setSample(0, 32, 0.7);

    // Simulate audio thread write
    pipeline_->process(buffer);

    // Simulate multiple UI thread reads (should all see 0.7)
    for (int i = 0; i < 100; ++i) {
        double peak = peakStorage_.load(std::memory_order_acquire);
        EXPECT_DOUBLE_EQ(peak, 0.7) << "Read " << i << " failed";
    }
}
