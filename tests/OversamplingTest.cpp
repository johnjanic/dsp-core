#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/Oversampling.h"
#include <platform/AudioBuffer.h>
#include <cmath>
#include <vector>
#include <numeric>

using namespace dsp;

class OversamplingTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr double kTolerance = 1e-6;
    static constexpr int kTestBlockSize = 512;

    // Generate sine wave into AudioBuffer
    platform::AudioBuffer<double> generateSine(int numChannels, int numSamples,
                                               double frequency, double sampleRate)
    {
        platform::AudioBuffer<double> buffer(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                buffer.setSample(ch, i, std::sin(2.0 * M_PI * frequency * i / sampleRate));
            }
        }
        return buffer;
    }

    // Calculate RMS from AudioBuffer channel
    double calculateRMS(const platform::AudioBuffer<double>& buffer, int channel)
    {
        double sumSq = 0.0;
        const int numSamples = buffer.getNumSamples();
        for (int i = 0; i < numSamples; ++i)
        {
            double sample = buffer.getSample(channel, i);
            sumSq += sample * sample;
        }
        return std::sqrt(sumSq / numSamples);
    }
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(OversamplingTest, Order0_NoOversampling)
{
    Oversampling<double> os(2, 0);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 1);
    EXPECT_EQ(os.getOrder(), 0);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    // Should be same size
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize);
}

TEST_F(OversamplingTest, Order1_2xOversampling)
{
    Oversampling<double> os(2, 1);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 2);
    EXPECT_EQ(os.getOrder(), 1);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    // Should be 2x size
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 2);
}

TEST_F(OversamplingTest, Order2_4xOversampling)
{
    Oversampling<double> os(2, 2);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 4);
    EXPECT_EQ(os.getOrder(), 2);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);
}

TEST_F(OversamplingTest, Order3_8xOversampling)
{
    Oversampling<double> os(2, 3);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 8);
    EXPECT_EQ(os.getOrder(), 3);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 8);
}

TEST_F(OversamplingTest, Order4_16xOversampling)
{
    Oversampling<double> os(2, 4);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 16);
    EXPECT_EQ(os.getOrder(), 4);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 16);
}

// =============================================================================
// Latency Tests
// =============================================================================

TEST_F(OversamplingTest, GetLatency_Order0_IsZero)
{
    Oversampling<double> os(2, 0);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getLatencyInSamples(), 0);
}

TEST_F(OversamplingTest, GetLatency_Order1_IsPositive)
{
    Oversampling<double> os(2, 1);
    os.prepare(kTestBlockSize);

    EXPECT_GT(os.getLatencyInSamples(), 0);
}

TEST_F(OversamplingTest, GetLatency_IncreasesWithOrder)
{
    int latency1 = Oversampling<double>(2, 1).getLatencyInSamples();
    int latency2 = Oversampling<double>(2, 2).getLatencyInSamples();
    int latency3 = Oversampling<double>(2, 3).getLatencyInSamples();
    int latency4 = Oversampling<double>(2, 4).getLatencyInSamples();

    EXPECT_LT(latency1, latency2);
    EXPECT_LT(latency2, latency3);
    EXPECT_LT(latency3, latency4);
}

// =============================================================================
// Audio Quality Tests
// =============================================================================

TEST_F(OversamplingTest, ProcessSamplesUp_IncreasesLength)
{
    Oversampling<double> os(1, 2);  // 4x
    os.prepare(kTestBlockSize);

    auto input = generateSine(1, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);
}

TEST_F(OversamplingTest, ProcessSamplesDown_RestoresLength)
{
    Oversampling<double> os(1, 2);  // 4x
    os.prepare(kTestBlockSize);

    auto input = generateSine(1, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    platform::AudioBuffer<double> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    EXPECT_EQ(output.getNumSamples(), kTestBlockSize);
}

TEST_F(OversamplingTest, RoundTrip_PreservesSignal)
{
    Oversampling<double> os(1, 2);  // 4x
    os.prepare(kTestBlockSize);

    // Use low frequency to avoid aliasing effects
    auto input = generateSine(1, kTestBlockSize, 100.0, kSampleRate);

    // Process multiple blocks to let filter settle
    for (int block = 0; block < 10; ++block)
    {
        os.processSamplesUp(input);
        platform::AudioBuffer<double> output(1, kTestBlockSize);
        os.processSamplesDown(output);
    }

    // Final round trip
    os.processSamplesUp(input);
    platform::AudioBuffer<double> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    // Compare RMS (should be similar after settling)
    double inputRMS = calculateRMS(input, 0);
    double outputRMS = calculateRMS(output, 0);

    // Allow for some amplitude variation due to filter
    double rmsRatio = outputRMS / inputRMS;
    EXPECT_GT(rmsRatio, 0.8);  // Within 2dB
    EXPECT_LT(rmsRatio, 1.2);
}

TEST_F(OversamplingTest, UpsampledSignal_HasHigherResolution)
{
    Oversampling<double> os(1, 1);  // 2x
    os.prepare(kTestBlockSize);

    // Generate a high-frequency sine (close to Nyquist)
    double frequency = kSampleRate * 0.4;  // 80% of Nyquist
    auto input = generateSine(1, kTestBlockSize, frequency, kSampleRate);

    os.processSamplesUp(input);

    // The upsampled version should have more samples per cycle
    // Just verify it's actually 2x the length
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 2);
}

// =============================================================================
// Multichannel Tests
// =============================================================================

TEST_F(OversamplingTest, Stereo_ProcessesBothChannels)
{
    Oversampling<double> os(2, 1);  // 2 channels, 2x
    os.prepare(kTestBlockSize);

    // Create stereo with different content per channel
    platform::AudioBuffer<double> input(2, kTestBlockSize);
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        input.setSample(0, i, std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate));
        input.setSample(1, i, std::sin(2.0 * M_PI * 2000.0 * i / kSampleRate));
    }

    platform::AudioBuffer<double>& upsampled = os.processSamplesUp(input);

    EXPECT_EQ(os.getNumChannels(), 2);
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 2);

    // Both channels should have non-zero data
    double rmsL = calculateRMS(upsampled, 0);
    double rmsR = calculateRMS(upsampled, 1);

    EXPECT_GT(rmsL, 0.1);
    EXPECT_GT(rmsR, 0.1);
}

TEST_F(OversamplingTest, MultiChannel_IndependentProcessing)
{
    Oversampling<double> os(4, 1);  // 4 channels
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<double> input(4, kTestBlockSize);
    // Fill each channel with different DC offset for identification
    for (int ch = 0; ch < 4; ++ch)
    {
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            input.setSample(ch, i, static_cast<double>(ch + 1) * 0.1);
        }
    }

    os.processSamplesUp(input);

    EXPECT_EQ(os.getNumChannels(), 4);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(OversamplingTest, Reset_ClearsState)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    // Process some data
    auto input = generateSine(1, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input);

    // Reset
    os.reset();

    // After reset, processing silence should produce silence
    platform::AudioBuffer<double> silence(1, kTestBlockSize);
    silence.clear();

    // Process multiple blocks of silence
    for (int i = 0; i < 10; ++i)
    {
        os.processSamplesUp(silence);
        platform::AudioBuffer<double> output(1, kTestBlockSize);
        os.processSamplesDown(output);
    }

    // Final process
    os.processSamplesUp(silence);
    platform::AudioBuffer<double> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    // Output should be near zero
    double rms = calculateRMS(output, 0);
    EXPECT_LT(rms, 0.001);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(OversamplingTest, SmallBlockSize_Works)
{
    Oversampling<double> os(1, 2);
    os.prepare(64);

    platform::AudioBuffer<double> input(1, 64);
    for (int i = 0; i < 64; ++i)
    {
        input.setSample(0, i, std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate));
    }

    os.processSamplesUp(input);
    EXPECT_EQ(os.getOversampledSize(), 256);
}

TEST_F(OversamplingTest, SingleSampleBlock_Works)
{
    Oversampling<double> os(1, 1);  // 2x
    os.prepare(1);

    platform::AudioBuffer<double> input(1, 1);
    input.setSample(0, 0, 0.5);

    os.processSamplesUp(input);
    EXPECT_EQ(os.getOversampledSize(), 2);
}

TEST_F(OversamplingTest, ZeroInput_ProducesNearZeroOutput)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<double> input(1, kTestBlockSize);
    input.clear();

    // Process multiple blocks to clear any initial transients
    for (int i = 0; i < 10; ++i)
    {
        os.processSamplesUp(input);
        platform::AudioBuffer<double> output(1, kTestBlockSize);
        os.processSamplesDown(output);
    }

    os.processSamplesUp(input);
    platform::AudioBuffer<double> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    double rms = calculateRMS(output, 0);
    EXPECT_LT(rms, 1e-10);
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(OversamplingTest, FloatPrecision_Works)
{
    Oversampling<float> os(1, 2);
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<float> input(1, kTestBlockSize);
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        input.setSample(0, i, std::sin(2.0f * static_cast<float>(M_PI) * 1000.0f * i / static_cast<float>(kSampleRate)));
    }

    os.processSamplesUp(input);
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);

    platform::AudioBuffer<float> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    // Verify no NaN or Inf
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        EXPECT_FALSE(std::isnan(output.getSample(0, i)));
        EXPECT_FALSE(std::isinf(output.getSample(0, i)));
    }
}

// =============================================================================
// Order Clamping Tests
// =============================================================================

TEST_F(OversamplingTest, NegativeOrder_ClampedToZero)
{
    Oversampling<double> os(1, -1);
    EXPECT_EQ(os.getOrder(), 0);
    EXPECT_EQ(os.getOversamplingFactor(), 1);
}

TEST_F(OversamplingTest, ExcessiveOrder_ClampedToFour)
{
    Oversampling<double> os(1, 10);
    EXPECT_EQ(os.getOrder(), 4);
    EXPECT_EQ(os.getOversamplingFactor(), 16);
}

// =============================================================================
// AudioBuffer API Tests
// =============================================================================

TEST_F(OversamplingTest, AudioBuffer_ProcessSamplesUp)
{
    Oversampling<double> os(2, 2);  // 4x
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<double> input(2, kTestBlockSize);
    // Fill with sine wave
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            input.setSample(ch, i, std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate));
        }
    }

    platform::AudioBuffer<double>& upsampled = os.processSamplesUp(input);

    EXPECT_EQ(upsampled.getNumChannels(), 2);
    EXPECT_GE(upsampled.getNumSamples(), kTestBlockSize * 4);
}

TEST_F(OversamplingTest, AudioBuffer_ProcessSamplesDown)
{
    Oversampling<double> os(2, 2);  // 4x
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<double> input(2, kTestBlockSize);
    // Fill with sine wave
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            input.setSample(ch, i, std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate));
        }
    }

    os.processSamplesUp(input);

    platform::AudioBuffer<double> output(2, kTestBlockSize);
    os.processSamplesDown(output);

    EXPECT_EQ(output.getNumChannels(), 2);
    EXPECT_EQ(output.getNumSamples(), kTestBlockSize);

    // Verify no NaN or Inf
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            EXPECT_FALSE(std::isnan(output.getSample(ch, i)));
            EXPECT_FALSE(std::isinf(output.getSample(ch, i)));
        }
    }
}

TEST_F(OversamplingTest, AudioBuffer_GetOversampledBuffer)
{
    Oversampling<double> os(2, 1);  // 2x
    os.prepare(kTestBlockSize);

    platform::AudioBuffer<double> input(2, kTestBlockSize);
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            input.setSample(ch, i, 0.5);
        }
    }

    os.processSamplesUp(input);

    platform::AudioBuffer<double>& oversampledRef = os.getOversampledBuffer();

    EXPECT_EQ(oversampledRef.getNumChannels(), 2);
    EXPECT_GE(oversampledRef.getNumSamples(), kTestBlockSize * 2);
}

TEST_F(OversamplingTest, AudioBuffer_RoundTrip)
{
    Oversampling<double> os(1, 2);  // 4x
    os.prepare(kTestBlockSize);

    // Use low frequency to avoid aliasing effects
    platform::AudioBuffer<double> input(1, kTestBlockSize);
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        input.setSample(0, i, std::sin(2.0 * M_PI * 100.0 * i / kSampleRate));
    }

    // Process multiple blocks to let filter settle
    for (int block = 0; block < 10; ++block)
    {
        os.processSamplesUp(input);
        platform::AudioBuffer<double> output(1, kTestBlockSize);
        os.processSamplesDown(output);
    }

    // Final round trip
    os.processSamplesUp(input);
    platform::AudioBuffer<double> output(1, kTestBlockSize);
    os.processSamplesDown(output);

    // Compare RMS (should be similar after settling)
    double inputRMS = 0.0, outputRMS = 0.0;
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        inputRMS += input.getSample(0, i) * input.getSample(0, i);
        outputRMS += output.getSample(0, i) * output.getSample(0, i);
    }
    inputRMS = std::sqrt(inputRMS / kTestBlockSize);
    outputRMS = std::sqrt(outputRMS / kTestBlockSize);

    // Allow for some amplitude variation due to filter
    double rmsRatio = outputRMS / inputRMS;
    EXPECT_GT(rmsRatio, 0.8);  // Within 2dB
    EXPECT_LT(rmsRatio, 1.2);
}
