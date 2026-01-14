#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/SmoothedValue.h"
#include <cmath>
#include <vector>

using namespace dsp;

class SmoothedValueTest : public ::testing::Test {
protected:
    static constexpr double kSampleRate = 44100.0;
    static constexpr double kRampTime = 0.01;  // 10ms
    static constexpr double kTolerance = 1e-9;
    static constexpr float kToleranceF = 1e-6f;
};

// =============================================================================
// Initialization Tests
// =============================================================================

TEST_F(SmoothedValueTest, DefaultConstruction_NotSmoothing)
{
    SmoothedValue<double> sv;
    EXPECT_FALSE(sv.isSmoothing());
}

TEST_F(SmoothedValueTest, ValueConstruction_SetsInitialValue)
{
    SmoothedValue<double> sv(0.5);
    EXPECT_NEAR(sv.getCurrentValue(), 0.5, kTolerance);
    EXPECT_NEAR(sv.getTargetValue(), 0.5, kTolerance);
    EXPECT_FALSE(sv.isSmoothing());
}

// =============================================================================
// SetCurrentAndTargetValue Tests
// =============================================================================

TEST_F(SmoothedValueTest, SetCurrentAndTarget_ImmediateValue)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);

    sv.setCurrentAndTargetValue(0.75);

    EXPECT_NEAR(sv.getCurrentValue(), 0.75, kTolerance);
    EXPECT_NEAR(sv.getTargetValue(), 0.75, kTolerance);
    EXPECT_NEAR(sv.getNextValue(), 0.75, kTolerance);
    EXPECT_FALSE(sv.isSmoothing());
}

TEST_F(SmoothedValueTest, SetCurrentAndTarget_StopsExistingSmoothing)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);  // Start smoothing

    EXPECT_TRUE(sv.isSmoothing());

    sv.setCurrentAndTargetValue(0.5);  // Interrupt

    EXPECT_FALSE(sv.isSmoothing());
    EXPECT_NEAR(sv.getCurrentValue(), 0.5, kTolerance);
}

// =============================================================================
// SetTargetValue Tests
// =============================================================================

TEST_F(SmoothedValueTest, SetTargetValue_StartsSmoothing)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);

    sv.setTargetValue(1.0);

    EXPECT_TRUE(sv.isSmoothing());
    EXPECT_NEAR(sv.getTargetValue(), 1.0, kTolerance);
}

TEST_F(SmoothedValueTest, SetTargetValue_SameValue_NoSmoothing)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.5);

    sv.setTargetValue(0.5);  // Same value

    EXPECT_FALSE(sv.isSmoothing());
}

TEST_F(SmoothedValueTest, SetTargetValue_ZeroRampTime_Immediate)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, 0.0);  // Zero ramp time
    sv.setCurrentAndTargetValue(0.0);

    sv.setTargetValue(1.0);

    EXPECT_FALSE(sv.isSmoothing());
    EXPECT_NEAR(sv.getCurrentValue(), 1.0, kTolerance);
}

// =============================================================================
// GetNextValue Tests
// =============================================================================

TEST_F(SmoothedValueTest, GetNextValue_ApproachesTarget)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    double firstValue = sv.getNextValue();
    double secondValue = sv.getNextValue();

    // Values should be increasing towards target
    EXPECT_GT(firstValue, 0.0);
    EXPECT_GT(secondValue, firstValue);
    EXPECT_LT(secondValue, 1.0);
}

TEST_F(SmoothedValueTest, GetNextValue_ReachesTargetEventually)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    // Ramp through all samples
    int expectedSteps = static_cast<int>(kRampTime * kSampleRate);
    for (int i = 0; i < expectedSteps + 10; ++i)
    {
        sv.getNextValue();
    }

    EXPECT_FALSE(sv.isSmoothing());
    EXPECT_NEAR(sv.getCurrentValue(), 1.0, kTolerance);
}

TEST_F(SmoothedValueTest, GetNextValue_NotSmoothing_ReturnsTarget)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.5);

    // Not smoothing, should return target
    EXPECT_NEAR(sv.getNextValue(), 0.5, kTolerance);
    EXPECT_NEAR(sv.getNextValue(), 0.5, kTolerance);
}

// =============================================================================
// Skip Tests
// =============================================================================

TEST_F(SmoothedValueTest, Skip_AdvancesMultipleSamples)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    int numSteps = static_cast<int>(kRampTime * kSampleRate);
    int skipAmount = numSteps / 2;

    double valueAfterSkip = sv.skip(skipAmount);

    // Should be approximately halfway to target
    EXPECT_GT(valueAfterSkip, 0.4);
    EXPECT_LT(valueAfterSkip, 0.6);
    EXPECT_TRUE(sv.isSmoothing());
}

TEST_F(SmoothedValueTest, Skip_PastEnd_SnapsToTarget)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    int numSteps = static_cast<int>(kRampTime * kSampleRate);

    double valueAfterSkip = sv.skip(numSteps + 100);

    EXPECT_NEAR(valueAfterSkip, 1.0, kTolerance);
    EXPECT_FALSE(sv.isSmoothing());
}

TEST_F(SmoothedValueTest, Skip_NotSmoothing_ReturnsTarget)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.5);

    double value = sv.skip(100);

    EXPECT_NEAR(value, 0.5, kTolerance);
}

// =============================================================================
// Reset Tests
// =============================================================================

TEST_F(SmoothedValueTest, Reset_ChangesRampTime)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, 0.1);  // 100ms
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    int numSteps1 = sv.getNumStepsRemaining();

    sv.setCurrentAndTargetValue(0.0);
    sv.reset(kSampleRate, 0.01);  // 10ms
    sv.setTargetValue(1.0);

    int numSteps2 = sv.getNumStepsRemaining();

    EXPECT_GT(numSteps1, numSteps2);
    EXPECT_NEAR(static_cast<double>(numSteps2) / numSteps1, 0.1, 0.01);
}

TEST_F(SmoothedValueTest, Reset_StopsSmoothing)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    EXPECT_TRUE(sv.isSmoothing());

    sv.reset(kSampleRate, kRampTime);

    EXPECT_FALSE(sv.isSmoothing());
}

// =============================================================================
// IsSmoothing Tests
// =============================================================================

TEST_F(SmoothedValueTest, IsSmoothing_TrueWhileRamping)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    int numSteps = static_cast<int>(kRampTime * kSampleRate);

    for (int i = 0; i < numSteps - 1; ++i)
    {
        EXPECT_TRUE(sv.isSmoothing()) << "Failed at step " << i;
        sv.getNextValue();
    }
}

TEST_F(SmoothedValueTest, IsSmoothing_FalseAtTarget)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    int numSteps = static_cast<int>(kRampTime * kSampleRate);

    for (int i = 0; i < numSteps + 1; ++i)
    {
        sv.getNextValue();
    }

    EXPECT_FALSE(sv.isSmoothing());
}

// =============================================================================
// Linear Ramp Accuracy Tests
// =============================================================================

TEST_F(SmoothedValueTest, LinearRamp_IsAccurate)
{
    SmoothedValue<double> sv;
    sv.reset(100);  // 100 steps (using int overload)
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(100.0);

    std::vector<double> values;
    for (int i = 0; i < 100; ++i)
    {
        values.push_back(sv.getNextValue());
    }

    // Each step should increase by 1.0
    for (int i = 1; i < 100; ++i)
    {
        double diff = values[i] - values[i-1];
        EXPECT_NEAR(diff, 1.0, kTolerance) << "Non-linear at step " << i;
    }

    // Final value should be target
    EXPECT_NEAR(values.back(), 100.0, kTolerance);
}

TEST_F(SmoothedValueTest, LinearRamp_DecreasingValues)
{
    SmoothedValue<double> sv;
    sv.reset(100);  // 100 steps (using int overload)
    sv.setCurrentAndTargetValue(100.0);
    sv.setTargetValue(0.0);

    for (int i = 0; i < 100; ++i)
    {
        double val = sv.getNextValue();
        double expected = 100.0 - static_cast<double>(i + 1);
        EXPECT_NEAR(val, expected, kTolerance) << "Failed at step " << i;
    }

    EXPECT_NEAR(sv.getCurrentValue(), 0.0, kTolerance);
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(SmoothedValueTest, FloatPrecision_Works)
{
    SmoothedValue<float> sv(0.0f);
    sv.reset(static_cast<double>(kSampleRate), kRampTime);
    sv.setTargetValue(1.0f);

    while (sv.isSmoothing())
    {
        sv.getNextValue();
    }

    EXPECT_NEAR(sv.getCurrentValue(), 1.0f, kToleranceF);
}

TEST_F(SmoothedValueTest, DoublePrecision_Works)
{
    SmoothedValue<double> sv(0.0);
    sv.reset(kSampleRate, kRampTime);
    sv.setTargetValue(1.0);

    while (sv.isSmoothing())
    {
        sv.getNextValue();
    }

    EXPECT_NEAR(sv.getCurrentValue(), 1.0, kTolerance);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(SmoothedValueTest, NegativeValues_WorkCorrectly)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, kRampTime);
    sv.setCurrentAndTargetValue(-1.0);
    sv.setTargetValue(1.0);

    while (sv.isSmoothing())
    {
        sv.getNextValue();
    }

    EXPECT_NEAR(sv.getCurrentValue(), 1.0, kTolerance);
}

TEST_F(SmoothedValueTest, VerySmallRampTime_Works)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, 0.0001);  // 0.1ms = ~4 samples
    sv.setCurrentAndTargetValue(0.0);
    sv.setTargetValue(1.0);

    EXPECT_TRUE(sv.isSmoothing());
    EXPECT_LE(sv.getNumStepsRemaining(), 10);

    while (sv.isSmoothing())
    {
        sv.getNextValue();
    }

    EXPECT_NEAR(sv.getCurrentValue(), 1.0, kTolerance);
}

TEST_F(SmoothedValueTest, MultipleTargetChanges_Works)
{
    SmoothedValue<double> sv;
    sv.reset(kSampleRate, 0.001);  // 1ms
    sv.setCurrentAndTargetValue(0.0);

    // Change target while smoothing
    sv.setTargetValue(1.0);
    for (int i = 0; i < 10; ++i) sv.getNextValue();

    sv.setTargetValue(0.5);  // New target
    while (sv.isSmoothing())
    {
        sv.getNextValue();
    }

    EXPECT_NEAR(sv.getCurrentValue(), 0.5, kTolerance);
}
