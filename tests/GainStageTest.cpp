#include <gtest/gtest.h>
#include "../dsp_core/Source/pipeline/GainStage.h"
#include <platform/AudioBuffer.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

class GainStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stage_.prepareToPlay(44100.0, 512);
    }

    GainStage stage_;
};

TEST_F(GainStageTest, DefaultGain_IsUnity) {
    platform::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, 1.0);
        }
    }

    stage_.process(buffer);

    // Should be unchanged (unity gain)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), 1.0, 1e-6)
                << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(GainStageTest, SetGainDecibels_AppliesCorrectly) {
    stage_.setGainDecibels(-6.0);
    stage_.reset(); // Snap to target (skip smoothing)

    platform::AudioBuffer<double> buffer(1, 64);
    for (int i = 0; i < 64; ++i) {
        buffer.setSample(0, i, 1.0);
    }

    stage_.process(buffer);

    double const expected = std::pow(10.0, -6.0 / 20.0);
    EXPECT_NEAR(buffer.getSample(0, 0), expected, 1e-5);
}

TEST_F(GainStageTest, SetGainLinear_AppliesCorrectly) {
    stage_.setGainLinear(0.5);
    stage_.reset(); // Snap to target

    platform::AudioBuffer<double> buffer(1, 64);
    for (int i = 0; i < 64; ++i) {
        buffer.setSample(0, i, 1.0);
    }

    stage_.process(buffer);

    EXPECT_NEAR(buffer.getSample(0, 0), 0.5, 1e-6);
}

TEST_F(GainStageTest, GetGainDecibels_ReturnsCorrectValue) {
    stage_.setGainDecibels(-12.0);
    EXPECT_NEAR(stage_.getGainDecibels(), -12.0, 1e-5);
}

TEST_F(GainStageTest, GetGainLinear_ReturnsCorrectValue) {
    stage_.setGainLinear(0.25);
    EXPECT_NEAR(stage_.getGainLinear(), 0.25, 1e-6);
}

TEST_F(GainStageTest, LegacySetGainDB_Works) {
    stage_.setGainDB(-6.0);
    stage_.reset();

    platform::AudioBuffer<double> buffer(1, 64);
    for (int i = 0; i < 64; ++i) {
        buffer.setSample(0, i, 1.0);
    }

    stage_.process(buffer);

    double const expected = std::pow(10.0, -6.0 / 20.0);
    EXPECT_NEAR(buffer.getSample(0, 0), expected, 1e-5);
}

TEST_F(GainStageTest, Smoothing_GradualChange) {
    // Start at unity gain
    stage_.reset();

    // Set to -20dB gain
    stage_.setGainDecibels(-20.0);

    platform::AudioBuffer<double> buffer(1, 64);
    for (int i = 0; i < 64; ++i) {
        buffer.setSample(0, i, 1.0);
    }

    stage_.process(buffer);

    // First sample should be close to 1.0 (starting value)
    // Last sample should be moving toward target
    double const firstSample = buffer.getSample(0, 0);
    double const lastSample = buffer.getSample(0, 63);
    double const target = std::pow(10.0, -20.0 / 20.0);

    // First sample should be close to 1.0
    EXPECT_GT(firstSample, 0.9);
    // Last sample should be between first and target
    EXPECT_LT(lastSample, firstSample);
}

TEST_F(GainStageTest, MultiChannel_ProcessesAllChannels) {
    stage_.setGainLinear(0.5);
    stage_.reset();

    platform::AudioBuffer<double> buffer(8, 64);
    for (int ch = 0; ch < 8; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, 1.0);
        }
    }

    stage_.process(buffer);

    for (int ch = 0; ch < 8; ++ch) {
        for (int i = 0; i < 64; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), 0.5, 1e-6)
                << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(GainStageTest, Reset_SnapsToTarget) {
    stage_.setGainLinear(0.1);
    stage_.reset(); // Snap to target

    platform::AudioBuffer<double> buffer(1, 64);
    for (int i = 0; i < 64; ++i) {
        buffer.setSample(0, i, 1.0);
    }

    stage_.process(buffer);

    // After reset, should apply immediately without smoothing
    EXPECT_NEAR(buffer.getSample(0, 0), 0.1, 1e-6);
    EXPECT_NEAR(buffer.getSample(0, 63), 0.1, 1e-6);
}

TEST_F(GainStageTest, GetName_ReturnsGain) {
    EXPECT_EQ(stage_.getName(), "Gain");
}

TEST_F(GainStageTest, GetLatencySamples_ReturnsZero) {
    EXPECT_EQ(stage_.getLatencySamples(), 0);
}
