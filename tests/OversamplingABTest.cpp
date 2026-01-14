#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/Oversampling.h"
#include <audio-primitives/AudioBuffer.h>
#include <cmath>
#include <vector>

using namespace dsp;

/**
 * @brief A/B comparison tests for oversampling.
 *
 * These tests compare the new implementation against reference data.
 * To generate reference data:
 * 1. Build with JUCE oversampling temporarily
 * 2. Run generate_reference_data test
 * 3. Save output to reference files
 */
class OversamplingABTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr int kTestBlockSize = 512;
    static constexpr double kMaxAcceptableDifference = 0.1;  // 10% tolerance initially

    // Helper struct for test buffers
    struct TestBuffer
    {
        std::vector<std::vector<double>> channels;
        std::vector<double*> channelPointers;
        std::vector<const double*> constChannelPointers;

        TestBuffer(int numChannels, int numSamples, double value = 0.0)
        {
            channels.resize(numChannels);
            channelPointers.resize(numChannels);
            constChannelPointers.resize(numChannels);
            for (int ch = 0; ch < numChannels; ++ch)
            {
                channels[ch].resize(numSamples, value);
                channelPointers[ch] = channels[ch].data();
                constChannelPointers[ch] = channels[ch].data();
            }
        }

        double** data() { return channelPointers.data(); }
        const double* const* constData() const { return constChannelPointers.data(); }
        double* getChannel(int ch) { return channels[ch].data(); }

        void clear()
        {
            for (auto& ch : channels)
            {
                std::fill(ch.begin(), ch.end(), 0.0);
            }
        }
    };

    // Generate test signals
    static TestBuffer generateSineSweep(int numSamples)
    {
        TestBuffer buffer(1, numSamples);
        double* data = buffer.getChannel(0);

        // Sweep from 20Hz to 20kHz
        for (int i = 0; i < numSamples; ++i)
        {
            double t = static_cast<double>(i) / kSampleRate;
            double freq = 20.0 * std::pow(1000.0, t / (numSamples / kSampleRate));
            data[i] = std::sin(2.0 * M_PI * freq * t);
        }
        return buffer;
    }

    static TestBuffer generateWhiteNoise(int numSamples)
    {
        TestBuffer buffer(1, numSamples);
        double* data = buffer.getChannel(0);

        // Simple LCG pseudo-random
        uint32_t seed = 12345;
        for (int i = 0; i < numSamples; ++i)
        {
            seed = seed * 1103515245 + 12345;
            data[i] = (static_cast<double>(seed) / UINT32_MAX) * 2.0 - 1.0;
        }
        return buffer;
    }

    static TestBuffer generateImpulse(int numSamples)
    {
        TestBuffer buffer(1, numSamples, 0.0);
        buffer.getChannel(0)[0] = 1.0;
        return buffer;
    }
};

// Note: These tests are placeholders. Full A/B comparison requires
// capturing reference data from JUCE implementation first.

TEST_F(OversamplingABTest, SineSweep_Produces_ValidOutput)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    auto input = generateSineSweep(kTestBlockSize);

    // Create AudioBuffer and copy input data
    audio::AudioBuffer<double> inputBuffer(1, kTestBlockSize);
    std::memcpy(inputBuffer.getWritePointer(0), input.getChannel(0), sizeof(double) * kTestBlockSize);

    os.processSamplesUp(inputBuffer);

    audio::AudioBuffer<double> outputBuffer(1, kTestBlockSize);
    os.processSamplesDown(outputBuffer);

    // Verify output is valid (not NaN/Inf)
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        EXPECT_FALSE(std::isnan(outputBuffer.getSample(0, i)));
        EXPECT_FALSE(std::isinf(outputBuffer.getSample(0, i)));
    }
}

TEST_F(OversamplingABTest, WhiteNoise_Produces_ValidOutput)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    auto input = generateWhiteNoise(kTestBlockSize);

    // Create AudioBuffer and copy input data
    audio::AudioBuffer<double> inputBuffer(1, kTestBlockSize);
    std::memcpy(inputBuffer.getWritePointer(0), input.getChannel(0), sizeof(double) * kTestBlockSize);

    os.processSamplesUp(inputBuffer);

    audio::AudioBuffer<double> outputBuffer(1, kTestBlockSize);
    os.processSamplesDown(outputBuffer);

    // Verify output is valid
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        EXPECT_FALSE(std::isnan(outputBuffer.getSample(0, i)));
        EXPECT_FALSE(std::isinf(outputBuffer.getSample(0, i)));
    }
}

TEST_F(OversamplingABTest, Impulse_Produces_ValidOutput)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    auto input = generateImpulse(kTestBlockSize);

    // Create AudioBuffer and copy input data
    audio::AudioBuffer<double> inputBuffer(1, kTestBlockSize);
    std::memcpy(inputBuffer.getWritePointer(0), input.getChannel(0), sizeof(double) * kTestBlockSize);

    os.processSamplesUp(inputBuffer);

    audio::AudioBuffer<double> outputBuffer(1, kTestBlockSize);
    os.processSamplesDown(outputBuffer);

    // Verify output is valid
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        EXPECT_FALSE(std::isnan(outputBuffer.getSample(0, i)));
        EXPECT_FALSE(std::isinf(outputBuffer.getSample(0, i)));
    }
}

// TODO: Add reference data comparison tests once JUCE reference is captured
// TEST_F(OversamplingABTest, SineSweep_Order1_MatchesJUCE)
// TEST_F(OversamplingABTest, SineSweep_Order2_MatchesJUCE)
// TEST_F(OversamplingABTest, DifferenceBelow_Minus120dB)
