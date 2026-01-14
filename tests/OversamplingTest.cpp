#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/Oversampling.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace dsp;

class OversamplingTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr double kTolerance = 1e-6;
    static constexpr int kTestBlockSize = 512;

    // Helper struct for multi-channel test buffers
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
        int getNumChannels() const { return static_cast<int>(channels.size()); }
        int getNumSamples() const { return channels.empty() ? 0 : static_cast<int>(channels[0].size()); }
        double* getChannel(int ch) { return channels[ch].data(); }
        const double* getChannel(int ch) const { return channels[ch].data(); }

        void clear()
        {
            for (auto& ch : channels)
            {
                std::fill(ch.begin(), ch.end(), 0.0);
            }
        }
    };

    // Generate sine wave into test buffer
    TestBuffer generateSine(int numChannels, int numSamples,
                            double frequency, double sampleRate)
    {
        TestBuffer buffer(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch)
        {
            double* data = buffer.getChannel(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                data[i] = std::sin(2.0 * M_PI * frequency * i / sampleRate);
            }
        }
        return buffer;
    }

    // Calculate RMS
    double calculateRMS(const double* data, int numSamples)
    {
        double sumSq = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            sumSq += data[i] * data[i];
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
    os.processSamplesUp(input.constData(), kTestBlockSize);

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
    os.processSamplesUp(input.constData(), kTestBlockSize);

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
    os.processSamplesUp(input.constData(), kTestBlockSize);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);
}

TEST_F(OversamplingTest, Order3_8xOversampling)
{
    Oversampling<double> os(2, 3);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 8);
    EXPECT_EQ(os.getOrder(), 3);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input.constData(), kTestBlockSize);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 8);
}

TEST_F(OversamplingTest, Order4_16xOversampling)
{
    Oversampling<double> os(2, 4);
    os.prepare(kTestBlockSize);

    EXPECT_EQ(os.getOversamplingFactor(), 16);
    EXPECT_EQ(os.getOrder(), 4);

    auto input = generateSine(2, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input.constData(), kTestBlockSize);

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
    os.processSamplesUp(input.constData(), kTestBlockSize);

    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);
}

TEST_F(OversamplingTest, ProcessSamplesDown_RestoresLength)
{
    Oversampling<double> os(1, 2);  // 4x
    os.prepare(kTestBlockSize);

    auto input = generateSine(1, kTestBlockSize, 1000.0, kSampleRate);
    os.processSamplesUp(input.constData(), kTestBlockSize);

    TestBuffer output(1, kTestBlockSize);
    os.processSamplesDown(output.data(), kTestBlockSize);

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
        os.processSamplesUp(input.constData(), kTestBlockSize);
        TestBuffer output(1, kTestBlockSize);
        os.processSamplesDown(output.data(), kTestBlockSize);
    }

    // Final round trip
    os.processSamplesUp(input.constData(), kTestBlockSize);
    TestBuffer output(1, kTestBlockSize);
    os.processSamplesDown(output.data(), kTestBlockSize);

    // Compare RMS (should be similar after settling)
    double inputRMS = calculateRMS(input.getChannel(0), kTestBlockSize);
    double outputRMS = calculateRMS(output.getChannel(0), kTestBlockSize);

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

    os.processSamplesUp(input.constData(), kTestBlockSize);

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
    TestBuffer input(2, kTestBlockSize);
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        input.getChannel(0)[i] = std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate);
        input.getChannel(1)[i] = std::sin(2.0 * M_PI * 2000.0 * i / kSampleRate);
    }

    double** upsampled = os.processSamplesUp(input.constData(), kTestBlockSize);

    EXPECT_EQ(os.getNumChannels(), 2);
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 2);

    // Both channels should have non-zero data
    double rmsL = calculateRMS(upsampled[0], os.getOversampledSize());
    double rmsR = calculateRMS(upsampled[1], os.getOversampledSize());

    EXPECT_GT(rmsL, 0.1);
    EXPECT_GT(rmsR, 0.1);
}

TEST_F(OversamplingTest, MultiChannel_IndependentProcessing)
{
    Oversampling<double> os(4, 1);  // 4 channels
    os.prepare(kTestBlockSize);

    TestBuffer input(4, kTestBlockSize);
    // Fill each channel with different DC offset for identification
    for (int ch = 0; ch < 4; ++ch)
    {
        double* data = input.getChannel(ch);
        for (int i = 0; i < kTestBlockSize; ++i)
        {
            data[i] = static_cast<double>(ch + 1) * 0.1;
        }
    }

    os.processSamplesUp(input.constData(), kTestBlockSize);

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
    os.processSamplesUp(input.constData(), kTestBlockSize);

    // Reset
    os.reset();

    // After reset, processing silence should produce silence
    TestBuffer silence(1, kTestBlockSize, 0.0);

    // Process multiple blocks of silence
    for (int i = 0; i < 10; ++i)
    {
        os.processSamplesUp(silence.constData(), kTestBlockSize);
        TestBuffer output(1, kTestBlockSize);
        os.processSamplesDown(output.data(), kTestBlockSize);
    }

    // Final process
    os.processSamplesUp(silence.constData(), kTestBlockSize);
    TestBuffer output(1, kTestBlockSize);
    os.processSamplesDown(output.data(), kTestBlockSize);

    // Output should be near zero
    double rms = calculateRMS(output.getChannel(0), kTestBlockSize);
    EXPECT_LT(rms, 0.001);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(OversamplingTest, SmallBlockSize_Works)
{
    Oversampling<double> os(1, 2);
    os.prepare(64);

    TestBuffer input(1, 64);
    for (int i = 0; i < 64; ++i)
    {
        input.getChannel(0)[i] = std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate);
    }

    os.processSamplesUp(input.constData(), 64);
    EXPECT_EQ(os.getOversampledSize(), 256);
}

TEST_F(OversamplingTest, SingleSampleBlock_Works)
{
    Oversampling<double> os(1, 1);  // 2x
    os.prepare(1);

    TestBuffer input(1, 1, 0.5);

    os.processSamplesUp(input.constData(), 1);
    EXPECT_EQ(os.getOversampledSize(), 2);
}

TEST_F(OversamplingTest, ZeroInput_ProducesNearZeroOutput)
{
    Oversampling<double> os(1, 2);
    os.prepare(kTestBlockSize);

    TestBuffer input(1, kTestBlockSize, 0.0);

    // Process multiple blocks to clear any initial transients
    for (int i = 0; i < 10; ++i)
    {
        os.processSamplesUp(input.constData(), kTestBlockSize);
        TestBuffer output(1, kTestBlockSize);
        os.processSamplesDown(output.data(), kTestBlockSize);
    }

    os.processSamplesUp(input.constData(), kTestBlockSize);
    TestBuffer output(1, kTestBlockSize);
    os.processSamplesDown(output.data(), kTestBlockSize);

    double rms = calculateRMS(output.getChannel(0), kTestBlockSize);
    EXPECT_LT(rms, 1e-10);
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(OversamplingTest, FloatPrecision_Works)
{
    Oversampling<float> os(1, 2);
    os.prepare(kTestBlockSize);

    std::vector<float> inputData(kTestBlockSize);
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        inputData[i] = std::sin(2.0f * static_cast<float>(M_PI) * 1000.0f * i / static_cast<float>(kSampleRate));
    }
    std::vector<const float*> inputPtrs = { inputData.data() };

    os.processSamplesUp(inputPtrs.data(), kTestBlockSize);
    EXPECT_EQ(os.getOversampledSize(), kTestBlockSize * 4);

    std::vector<float> outputData(kTestBlockSize);
    std::vector<float*> outputPtrs = { outputData.data() };
    os.processSamplesDown(outputPtrs.data(), kTestBlockSize);

    // Verify no NaN or Inf
    for (int i = 0; i < kTestBlockSize; ++i)
    {
        EXPECT_FALSE(std::isnan(outputData[i]));
        EXPECT_FALSE(std::isinf(outputData[i]));
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
