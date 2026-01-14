#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/Gain.h"
#include <cmath>
#include <vector>

using namespace dsp;

class GainTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr double kTolerance = 1e-9;
    static constexpr float kToleranceF = 1e-6f;

    // Create test buffer with known values (multi-channel)
    struct TestBuffer
    {
        std::vector<std::vector<double>> channels;
        std::vector<double*> channelPointers;

        TestBuffer(int numChannels, int numSamples, double value = 1.0)
        {
            channels.resize(numChannels);
            channelPointers.resize(numChannels);
            for (int ch = 0; ch < numChannels; ++ch)
            {
                channels[ch].resize(numSamples, value);
                channelPointers[ch] = channels[ch].data();
            }
        }

        double** data() { return channelPointers.data(); }
        int getNumChannels() const { return static_cast<int>(channels.size()); }
        int getNumSamples() const { return channels.empty() ? 0 : static_cast<int>(channels[0].size()); }
        double* getChannel(int ch) { return channels[ch].data(); }
        const double* getChannel(int ch) const { return channels[ch].data(); }
    };
};

// =============================================================================
// Initialization Tests
// =============================================================================

TEST_F(GainTest, DefaultGain_IsUnity)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);

    EXPECT_NEAR(gain.getGainLinear(), 1.0, kTolerance);
    EXPECT_NEAR(gain.getGainDecibels(), 0.0, kTolerance);
}

TEST_F(GainTest, AfterPrepare_NotSmoothing)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);

    EXPECT_FALSE(gain.isSmoothing());
}

// =============================================================================
// Gain Setting Tests
// =============================================================================

TEST_F(GainTest, SetGainLinear_AppliesToBuffer)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.5);

    // Skip smoothing by resetting
    gain.reset();

    TestBuffer buffer(2, 64, 1.0);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    // All samples should be 0.5
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const double* data = buffer.getChannel(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i)
        {
            EXPECT_NEAR(data[i], 0.5, kTolerance);
        }
    }
}

TEST_F(GainTest, SetGainDecibels_ConvertsCorrectly)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);

    gain.setGainDecibels(-6.0);
    double expectedGain = std::pow(10.0, -6.0 / 20.0);
    EXPECT_NEAR(gain.getGainLinear(), expectedGain, kTolerance);

    gain.setGainDecibels(0.0);
    EXPECT_NEAR(gain.getGainLinear(), 1.0, kTolerance);

    gain.setGainDecibels(6.0);
    expectedGain = std::pow(10.0, 6.0 / 20.0);
    EXPECT_NEAR(gain.getGainLinear(), expectedGain, kTolerance);
}

TEST_F(GainTest, ZeroGain_SilencesOutput)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.0);
    gain.reset();

    TestBuffer buffer(2, 64, 1.0);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const double* data = buffer.getChannel(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i)
        {
            EXPECT_NEAR(data[i], 0.0, kTolerance);
        }
    }
}

TEST_F(GainTest, NegativeDb_AttenuatesSignal)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainDecibels(-20.0);  // 0.1 linear
    gain.reset();

    TestBuffer buffer(1, 64, 1.0);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    const double* data = buffer.getChannel(0);
    EXPECT_NEAR(data[0], 0.1, kTolerance);
}

TEST_F(GainTest, PositiveDb_BoostsSignal)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainDecibels(20.0);  // 10.0 linear
    gain.reset();

    TestBuffer buffer(1, 64, 1.0);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    const double* data = buffer.getChannel(0);
    EXPECT_NEAR(data[0], 10.0, kTolerance);
}

// =============================================================================
// Smoothing Tests
// =============================================================================

TEST_F(GainTest, Process_SmoothsChanges)
{
    Gain<double> gain;
    gain.prepare(kSampleRate, 0.01);  // 10ms ramp
    gain.setGainLinear(0.5);

    EXPECT_TRUE(gain.isSmoothing());

    // Process until smoothing completes
    int maxIterations = 1000;
    while (gain.isSmoothing() && maxIterations-- > 0)
    {
        TestBuffer buffer(1, 64, 1.0);
        gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());
    }

    EXPECT_FALSE(gain.isSmoothing());
}

TEST_F(GainTest, Smoothing_GradualChange)
{
    Gain<double> gain;
    gain.prepare(kSampleRate, 0.001);  // 1ms ramp (~44 samples)
    gain.setGainLinear(1.0);
    gain.reset();

    // Now change to 0.5
    gain.setGainLinear(0.5);

    std::vector<double> outputs;
    for (int i = 0; i < 100; ++i)
    {
        outputs.push_back(gain.processSample(1.0));
    }

    // First sample should be close to 1.0
    EXPECT_GT(outputs[0], 0.9);

    // Middle samples should be between
    bool foundMiddle = false;
    for (double out : outputs)
    {
        if (out > 0.55 && out < 0.95)
        {
            foundMiddle = true;
            break;
        }
    }
    EXPECT_TRUE(foundMiddle);

    // Last sample should be close to 0.5
    EXPECT_NEAR(outputs.back(), 0.5, 0.01);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(GainTest, Reset_ClearsState)
{
    Gain<double> gain;
    gain.prepare(kSampleRate, 0.1);  // Long ramp
    gain.setGainLinear(0.5);

    EXPECT_TRUE(gain.isSmoothing());

    gain.reset();

    EXPECT_FALSE(gain.isSmoothing());

    // First processed sample should be at target
    double output = gain.processSample(1.0);
    EXPECT_NEAR(output, 0.5, kTolerance);
}

TEST_F(GainTest, Prepare_SetsRampTime)
{
    Gain<double> gain;

    // Short ramp
    gain.prepare(kSampleRate, 0.001);  // 1ms
    gain.setGainLinear(1.0);
    gain.reset();
    gain.setGainLinear(0.5);

    int shortRampSamples = 0;
    while (gain.isSmoothing())
    {
        (void)gain.processSample(1.0);
        shortRampSamples++;
        if (shortRampSamples > 1000) break;
    }

    // Long ramp
    gain.prepare(kSampleRate, 0.01);  // 10ms
    gain.setGainLinear(1.0);
    gain.reset();
    gain.setGainLinear(0.5);

    int longRampSamples = 0;
    while (gain.isSmoothing())
    {
        (void)gain.processSample(1.0);
        longRampSamples++;
        if (longRampSamples > 10000) break;
    }

    // Long ramp should take ~10x more samples
    EXPECT_GT(longRampSamples, shortRampSamples * 5);
}

// =============================================================================
// Processing Tests
// =============================================================================

TEST_F(GainTest, ProcessSample_Works)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.5);
    gain.reset();

    double output = gain.processSample(1.0);
    EXPECT_NEAR(output, 0.5, kTolerance);

    output = gain.processSample(2.0);
    EXPECT_NEAR(output, 1.0, kTolerance);
}

TEST_F(GainTest, ProcessBlock_Works)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.5);
    gain.reset();

    std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    gain.processBlock(samples.data(), static_cast<int>(samples.size()));

    EXPECT_NEAR(samples[0], 0.5, kTolerance);
    EXPECT_NEAR(samples[1], 1.0, kTolerance);
    EXPECT_NEAR(samples[2], 1.5, kTolerance);
    EXPECT_NEAR(samples[3], 2.0, kTolerance);
    EXPECT_NEAR(samples[4], 2.5, kTolerance);
}

TEST_F(GainTest, ProcessBlock_MultiChannel)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.5);
    gain.reset();

    TestBuffer buffer(4, 64, 1.0);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    for (int ch = 0; ch < 4; ++ch)
    {
        const double* data = buffer.getChannel(ch);
        for (int i = 0; i < 64; ++i)
        {
            EXPECT_NEAR(data[i], 0.5, kTolerance) << "Channel " << ch << " sample " << i;
        }
    }
}

TEST_F(GainTest, UnityGain_NoChange)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(1.0);
    gain.reset();

    TestBuffer buffer(2, 64, 0.75);
    gain.processBlock(buffer.data(), buffer.getNumChannels(), buffer.getNumSamples());

    const double* data = buffer.getChannel(0);
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_NEAR(data[i], 0.75, kTolerance);
    }
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(GainTest, FloatPrecision_Works)
{
    Gain<float> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(0.5f);
    gain.reset();

    std::vector<float> samples(64, 1.0f);
    gain.processBlock(samples.data(), static_cast<int>(samples.size()));

    EXPECT_NEAR(samples[0], 0.5f, kToleranceF);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(GainTest, VerySmallGain_Works)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(1e-10);
    gain.reset();

    double output = gain.processSample(1.0);
    EXPECT_NEAR(output, 1e-10, 1e-20);
}

TEST_F(GainTest, LargeGain_Works)
{
    Gain<double> gain;
    gain.prepare(kSampleRate);
    gain.setGainLinear(1000.0);
    gain.reset();

    double output = gain.processSample(1.0);
    EXPECT_NEAR(output, 1000.0, kTolerance);
}
