#include <gtest/gtest.h>
#include "dsp_core/Source/primitives/Decibels.h"
#include <cmath>
#include <limits>

using namespace dsp;

class DecibelsTest : public ::testing::Test {
protected:
    // Tolerance for floating point comparisons
    static constexpr double kTolerance = 1e-9;
    static constexpr float kToleranceF = 1e-6f;
};

// =============================================================================
// dB to Gain Conversion Tests
// =============================================================================

TEST_F(DecibelsTest, ZeroDb_IsUnityGain)
{
    EXPECT_NEAR(Decibels::decibelsToGain(0.0), 1.0, kTolerance);
    EXPECT_NEAR(Decibels::decibelsToGain(0.0f), 1.0f, kToleranceF);
}

TEST_F(DecibelsTest, Minus6Db_IsApproximatelyHalfGain)
{
    // -6.0206 dB is exactly half, -6 dB is close
    const double expected = std::pow(10.0, -6.0 / 20.0);  // ~0.501187
    EXPECT_NEAR(Decibels::decibelsToGain(-6.0), expected, kTolerance);
}

TEST_F(DecibelsTest, Plus6Db_IsApproximatelyDoubleGain)
{
    const double expected = std::pow(10.0, 6.0 / 20.0);  // ~1.99526
    EXPECT_NEAR(Decibels::decibelsToGain(6.0), expected, kTolerance);
}

TEST_F(DecibelsTest, Minus20Db_IsTenthGain)
{
    // -20 dB is exactly 0.1 gain
    EXPECT_NEAR(Decibels::decibelsToGain(-20.0), 0.1, kTolerance);
}

TEST_F(DecibelsTest, Plus20Db_IsTenTimesGain)
{
    // +20 dB is exactly 10.0 gain
    EXPECT_NEAR(Decibels::decibelsToGain(20.0), 10.0, kTolerance);
}

TEST_F(DecibelsTest, VeryLowDb_ReturnsZero)
{
    // Values at or below -100 dB should return 0
    EXPECT_EQ(Decibels::decibelsToGain(-100.0), 0.0);
    EXPECT_EQ(Decibels::decibelsToGain(-120.0), 0.0);
    EXPECT_EQ(Decibels::decibelsToGain(-200.0), 0.0);
}

TEST_F(DecibelsTest, CustomFloor_RespectsThreshold)
{
    // With -60 dB floor, -60 dB and below return 0
    EXPECT_EQ(Decibels::decibelsToGain(-60.0, -60.0), 0.0);
    EXPECT_EQ(Decibels::decibelsToGain(-70.0, -60.0), 0.0);
    // Just above floor should return non-zero
    EXPECT_GT(Decibels::decibelsToGain(-59.9, -60.0), 0.0);
}

// =============================================================================
// Gain to dB Conversion Tests
// =============================================================================

TEST_F(DecibelsTest, UnityGain_IsZeroDb)
{
    EXPECT_NEAR(Decibels::gainToDecibels(1.0), 0.0, kTolerance);
    EXPECT_NEAR(Decibels::gainToDecibels(1.0f), 0.0f, kToleranceF);
}

TEST_F(DecibelsTest, HalfGain_IsApproximatelyMinus6Db)
{
    const double expected = 20.0 * std::log10(0.5);  // ~-6.0206
    EXPECT_NEAR(Decibels::gainToDecibels(0.5), expected, kTolerance);
}

TEST_F(DecibelsTest, DoubleGain_IsApproximatelyPlus6Db)
{
    const double expected = 20.0 * std::log10(2.0);  // ~6.0206
    EXPECT_NEAR(Decibels::gainToDecibels(2.0), expected, kTolerance);
}

TEST_F(DecibelsTest, TenthGain_IsMinus20Db)
{
    EXPECT_NEAR(Decibels::gainToDecibels(0.1), -20.0, kTolerance);
}

TEST_F(DecibelsTest, TenTimesGain_IsPlus20Db)
{
    EXPECT_NEAR(Decibels::gainToDecibels(10.0), 20.0, kTolerance);
}

TEST_F(DecibelsTest, ZeroGain_ReturnsMinusInfinity)
{
    // Zero gain should return the floor (-100 dB default)
    EXPECT_EQ(Decibels::gainToDecibels(0.0), -100.0);
}

TEST_F(DecibelsTest, NegativeGain_ReturnsMinusInfinity)
{
    // Negative gain is invalid, treated as zero
    EXPECT_EQ(Decibels::gainToDecibels(-1.0), -100.0);
}

TEST_F(DecibelsTest, VerySmallGain_ClampsToFloor)
{
    // Gains resulting in < -100 dB should clamp
    double verySmall = std::pow(10.0, -110.0 / 20.0);
    EXPECT_EQ(Decibels::gainToDecibels(verySmall), -100.0);
}

TEST_F(DecibelsTest, CustomFloor_ClampsCorrectly)
{
    // With -60 dB floor
    double small = std::pow(10.0, -70.0 / 20.0);
    EXPECT_EQ(Decibels::gainToDecibels(small, -60.0), -60.0);
}

// =============================================================================
// Round-Trip Tests
// =============================================================================

TEST_F(DecibelsTest, RoundTrip_DbToGainToDb)
{
    // Test various dB values survive round-trip
    for (double db = -90.0; db <= 24.0; db += 6.0)
    {
        double gain = Decibels::decibelsToGain(db);
        double dbBack = Decibels::gainToDecibels(gain);
        EXPECT_NEAR(dbBack, db, kTolerance) << "Failed at " << db << " dB";
    }
}

TEST_F(DecibelsTest, RoundTrip_GainToDbToGain)
{
    // Test various gain values survive round-trip
    for (double gain = 0.001; gain <= 10.0; gain *= 2.0)
    {
        double db = Decibels::gainToDecibels(gain);
        double gainBack = Decibels::decibelsToGain(db);
        EXPECT_NEAR(gainBack, gain, kTolerance * gain) << "Failed at gain " << gain;
    }
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(DecibelsTest, FloatPrecision_Works)
{
    float dbF = -12.0f;
    float gainF = Decibels::decibelsToGain(dbF);
    float dbBackF = Decibels::gainToDecibels(gainF);
    EXPECT_NEAR(dbBackF, dbF, kToleranceF);
}

TEST_F(DecibelsTest, DoublePrecision_Works)
{
    double dbD = -12.0;
    double gainD = Decibels::decibelsToGain(dbD);
    double dbBackD = Decibels::gainToDecibels(gainD);
    EXPECT_NEAR(dbBackD, dbD, kTolerance);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(DecibelsTest, LargePositiveDb_HandledCorrectly)
{
    // +60 dB = 1000x gain
    EXPECT_NEAR(Decibels::decibelsToGain(60.0), 1000.0, kTolerance);
}

TEST_F(DecibelsTest, ExtremelyLargeGain_HandledCorrectly)
{
    // 1000x gain = +60 dB
    EXPECT_NEAR(Decibels::gainToDecibels(1000.0), 60.0, kTolerance);
}
