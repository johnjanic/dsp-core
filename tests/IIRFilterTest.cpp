#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/IIRFilter.h"
#include <platform/AudioBuffer.h>
#include <cmath>
#include <vector>

using namespace dsp;

class IIRFilterTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr double kTolerance = 1e-6;

    // Generate sine wave test signal
    std::vector<double> generateSine(double frequency, int numSamples)
    {
        std::vector<double> signal(numSamples);
        for (int i = 0; i < numSamples; ++i)
        {
            signal[i] = std::sin(2.0 * M_PI * frequency * i / kSampleRate);
        }
        return signal;
    }

    // Calculate RMS of a signal
    double calculateRMS(const std::vector<double>& signal)
    {
        double sumSq = 0.0;
        for (double s : signal)
        {
            sumSq += s * s;
        }
        return std::sqrt(sumSq / signal.size());
    }
};

// =============================================================================
// Coefficient Generation Tests
// =============================================================================

TEST_F(IIRFilterTest, DefaultCoefficients_PassThrough)
{
    IIRFilter<double> filter;
    auto coeffs = filter.getCoefficients();

    // Default should be pass-through (b0=1, others=0)
    EXPECT_NEAR(coeffs.b0, 1.0, kTolerance);
    EXPECT_NEAR(coeffs.b1, 0.0, kTolerance);
    EXPECT_NEAR(coeffs.b2, 0.0, kTolerance);
    EXPECT_NEAR(coeffs.a1, 0.0, kTolerance);
    EXPECT_NEAR(coeffs.a2, 0.0, kTolerance);
}

TEST_F(IIRFilterTest, Coefficients_MakeHighPass_ValidCoeffs)
{
    auto coeffs = IIRCoefficients<double>::makeHighPass(kSampleRate, 1000.0);

    // High-pass: b0 should be positive, b0 == b2, b1 < 0
    EXPECT_GT(coeffs.b0, 0.0);
    EXPECT_NEAR(coeffs.b0, coeffs.b2, kTolerance);
    EXPECT_LT(coeffs.b1, 0.0);
}

TEST_F(IIRFilterTest, Coefficients_MakeLowPass_ValidCoeffs)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);

    // Low-pass: all b coefficients should be positive
    EXPECT_GT(coeffs.b0, 0.0);
    EXPECT_GT(coeffs.b1, 0.0);
    EXPECT_GT(coeffs.b2, 0.0);
    EXPECT_NEAR(coeffs.b0, coeffs.b2, kTolerance);
}

TEST_F(IIRFilterTest, Coefficients_MakeFirstOrderHighPass_ValidCoeffs)
{
    auto coeffs = IIRCoefficients<double>::makeFirstOrderHighPass(kSampleRate, 100.0);

    // First-order: b2 and a2 should be zero
    EXPECT_NEAR(coeffs.b2, 0.0, kTolerance);
    EXPECT_NEAR(coeffs.a2, 0.0, kTolerance);
}

// =============================================================================
// High-Pass Filter Tests
// =============================================================================

TEST_F(IIRFilterTest, HighPass_AttenuatesLowFreq)
{
    auto coeffs = IIRCoefficients<double>::makeHighPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Generate 100 Hz signal (below cutoff)
    auto input = generateSine(100.0, 4096);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 1000, output.end()));

    // Output should be significantly attenuated (< -20 dB)
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_LT(gainDb, -20.0);
}

TEST_F(IIRFilterTest, HighPass_PassesHighFreq)
{
    auto coeffs = IIRCoefficients<double>::makeHighPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Generate 10000 Hz signal (above cutoff)
    auto input = generateSine(10000.0, 4096);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 1000, output.end()));

    // Output should be close to input (within -3 dB)
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_GT(gainDb, -3.0);
}

// =============================================================================
// Low-Pass Filter Tests
// =============================================================================

TEST_F(IIRFilterTest, LowPass_AttenuatesHighFreq)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Generate 10000 Hz signal (above cutoff)
    auto input = generateSine(10000.0, 4096);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 1000, output.end()));

    // Output should be significantly attenuated
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_LT(gainDb, -20.0);
}

TEST_F(IIRFilterTest, LowPass_PassesLowFreq)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Generate 100 Hz signal (below cutoff)
    auto input = generateSine(100.0, 4096);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 1000, output.end()));

    // Output should be close to input (within -1 dB)
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_GT(gainDb, -1.0);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(IIRFilterTest, Reset_ClearsState)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Process some samples to fill state
    for (int i = 0; i < 100; ++i)
    {
        filter.processSample(1.0);
    }

    // Reset
    filter.reset();

    // First output after reset should be same as fresh filter
    IIRFilter<double> freshFilter(coeffs);
    double testInput = 0.5;

    EXPECT_NEAR(filter.processSample(testInput),
                freshFilter.processSample(testInput),
                kTolerance);
}

// =============================================================================
// Processing Tests
// =============================================================================

TEST_F(IIRFilterTest, ProcessSample_SingleSample)
{
    IIRFilter<double> filter;  // Pass-through
    double input = 0.5;
    double output = filter.processSample(input);
    EXPECT_NEAR(output, input, kTolerance);
}

TEST_F(IIRFilterTest, ProcessBlock_MultiSample)
{
    IIRFilter<double> filter;  // Pass-through
    std::vector<double> input = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    for (size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_NEAR(output[i], input[i], kTolerance);
    }
}

TEST_F(IIRFilterTest, ProcessBlock_InPlace)
{
    IIRFilter<double> filter;  // Pass-through
    std::vector<double> samples = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> expected = samples;

    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = filter.processSample(samples[i]);

    for (size_t i = 0; i < samples.size(); ++i)
    {
        EXPECT_NEAR(samples[i], expected[i], kTolerance);
    }
}

// =============================================================================
// Frequency Response Tests
// =============================================================================

TEST_F(IIRFilterTest, FrequencyResponse_HighPassAtCutoff)
{
    double cutoff = 1000.0;
    auto coeffs = IIRCoefficients<double>::makeHighPass(kSampleRate, cutoff);
    IIRFilter<double> filter(coeffs);

    // Generate signal at cutoff frequency
    auto input = generateSine(cutoff, 8192);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 2000, output.end()));

    // At cutoff, gain should be approximately -3 dB
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_NEAR(gainDb, -3.0, 1.0);  // Within 1 dB
}

TEST_F(IIRFilterTest, FrequencyResponse_LowPassAtCutoff)
{
    double cutoff = 1000.0;
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, cutoff);
    IIRFilter<double> filter(coeffs);

    // Generate signal at cutoff frequency
    auto input = generateSine(cutoff, 8192);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 2000, output.end()));

    // At cutoff, gain should be approximately -3 dB
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_NEAR(gainDb, -3.0, 1.0);  // Within 1 dB
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(IIRFilterTest, FloatPrecision_Works)
{
    auto coeffs = IIRCoefficients<float>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<float> filter(coeffs);

    float input = 0.5f;
    float output = filter.processSample(input);

    EXPECT_FALSE(std::isnan(output));
    EXPECT_FALSE(std::isinf(output));
}

TEST_F(IIRFilterTest, DoublePrecision_Works)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    double input = 0.5;
    double output = filter.processSample(input);

    EXPECT_FALSE(std::isnan(output));
    EXPECT_FALSE(std::isinf(output));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(IIRFilterTest, VeryLowCutoff_Works)
{
    auto coeffs = IIRCoefficients<double>::makeHighPass(kSampleRate, 10.0);
    IIRFilter<double> filter(coeffs);

    // Should not produce NaN or Inf
    for (int i = 0; i < 1000; ++i)
    {
        double output = filter.processSample(0.5);
        EXPECT_FALSE(std::isnan(output));
        EXPECT_FALSE(std::isinf(output));
    }
}

TEST_F(IIRFilterTest, HighCutoff_Works)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 20000.0);
    IIRFilter<double> filter(coeffs);

    // Should not produce NaN or Inf
    for (int i = 0; i < 1000; ++i)
    {
        double output = filter.processSample(0.5);
        EXPECT_FALSE(std::isnan(output));
        EXPECT_FALSE(std::isinf(output));
    }
}

TEST_F(IIRFilterTest, ZeroInput_ProducesZeroOutput)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Process zeros
    for (int i = 0; i < 10000; ++i)
    {
        filter.processSample(0.0);
    }

    // After many zeros, output should be essentially zero
    double output = filter.processSample(0.0);
    EXPECT_NEAR(output, 0.0, 1e-15);
}

TEST_F(IIRFilterTest, ImpulseResponse_Decays)
{
    auto coeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 1000.0);
    IIRFilter<double> filter(coeffs);

    // Send impulse
    filter.processSample(1.0);

    // Track peak and final value
    double peakOutput = 0.0;
    double lastOutput = 0.0;

    // Send zeros - impulse response will oscillate then decay
    for (int i = 0; i < 1000; ++i)
    {
        double output = filter.processSample(0.0);
        peakOutput = std::max(peakOutput, std::abs(output));
        lastOutput = output;
    }

    // Peak should be bounded (stable filter)
    EXPECT_LT(peakOutput, 1.5);

    // Should have decayed significantly after 1000 samples
    EXPECT_LT(std::abs(lastOutput), 0.001);
}

// =============================================================================
// Band-Pass and Notch Filter Tests
// =============================================================================

TEST_F(IIRFilterTest, BandPass_PassesCenterFreq)
{
    double centerFreq = 1000.0;
    auto coeffs = IIRCoefficients<double>::makeBandPass(kSampleRate, centerFreq, 1.0);
    IIRFilter<double> filter(coeffs);

    // Generate signal at center frequency
    auto input = generateSine(centerFreq, 8192);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 2000, output.end()));

    // At center frequency, signal should pass through with minimal attenuation
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_GT(gainDb, -3.0);  // Within 3 dB
}

TEST_F(IIRFilterTest, Notch_AttenuatesCenterFreq)
{
    double centerFreq = 1000.0;
    auto coeffs = IIRCoefficients<double>::makeNotch(kSampleRate, centerFreq, 10.0);
    IIRFilter<double> filter(coeffs);

    // Generate signal at center frequency
    auto input = generateSine(centerFreq, 8192);
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 2000, output.end()));

    // At notch frequency, signal should be significantly attenuated
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_LT(gainDb, -20.0);
}

// =============================================================================
// SetCoefficients Tests
// =============================================================================

TEST_F(IIRFilterTest, SetCoefficients_ChangesFilterBehavior)
{
    IIRFilter<double> filter;

    // Start with pass-through, verify it passes
    double output1 = filter.processSample(1.0);
    EXPECT_NEAR(output1, 1.0, kTolerance);

    filter.reset();

    // Change to low-pass filter
    auto lpCoeffs = IIRCoefficients<double>::makeLowPass(kSampleRate, 100.0);
    filter.setCoefficients(lpCoeffs);
    filter.reset();

    // Now verify it filters
    auto input = generateSine(10000.0, 4096);
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        output[i] = filter.processSample(input[i]);

    double inputRMS = calculateRMS(input);
    double outputRMS = calculateRMS(std::vector<double>(output.begin() + 1000, output.end()));

    // High frequency should be attenuated
    double gainDb = 20.0 * std::log10(outputRMS / inputRMS);
    EXPECT_LT(gainDb, -20.0);
}

// =============================================================================
// AudioBuffer API Tests
// =============================================================================

TEST_F(IIRFilterTest, ProcessBlock_AudioBuffer_SingleChannel)
{
    IIRFilter<double> filter;  // Pass-through

    platform::AudioBuffer<double> buffer(2, 64);
    // Fill channel 0 with 0.5
    for (int i = 0; i < 64; ++i)
    {
        buffer.setSample(0, i, 0.5);
        buffer.setSample(1, i, 0.25);
    }

    // Process only channel 0
    filter.processBlock(buffer, 0);

    // Channel 0 should be unchanged (pass-through)
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_NEAR(buffer.getSample(0, i), 0.5, kTolerance);
    }

    // Channel 1 should be untouched
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_NEAR(buffer.getSample(1, i), 0.25, kTolerance);
    }
}
