#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for CurveFeatureDetector service
 * Tests geometric feature detection for spline anchor placement
 */
class CurveFeatureDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
    }

    // Helper: Create tanh curve (monotonic with inflection)
    void createTanhCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            double y = std::tanh(3.0 * x);
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Helper: Create sine curve (multiple extrema)
    void createSineCurve(int numPeriods = 2) {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            double y = std::sin(numPeriods * M_PI * x);
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Helper: Create cubic curve (monotonic with inflection at x=0)
    void createCubicCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            double y = x * x * x;
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Helper: Create linear curve (no features)
    void createLinearCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            ltf->setBaseLayerValue(i, x);
        }
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

// ============================================================================
// Basic Feature Detection Tests
// ============================================================================

TEST_F(CurveFeatureDetectorTest, DetectsNoExtremaInMonotonicTanh) {
    createTanhCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    EXPECT_EQ(0, features.localExtrema.size()) << "Tanh is monotonic, should have no local extrema";
    EXPECT_GE(features.inflectionPoints.size(), 1) << "Tanh has one inflection point near center";
}

TEST_F(CurveFeatureDetectorTest, DetectsSineExtrema) {
    createSineCurve(2);  // 2 full periods
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // 2 periods of sine have 4 extrema (2 peaks + 2 valleys)
    EXPECT_GE(features.localExtrema.size(), 3) << "2 periods of sine should have ~4 extrema";
    EXPECT_LE(features.localExtrema.size(), 5) << "Should not detect spurious extrema";
}

TEST_F(CurveFeatureDetectorTest, DetectsCubicInflection) {
    createCubicCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    EXPECT_EQ(0, features.localExtrema.size()) << "x³ is monotonic, no local extrema";
    EXPECT_GE(features.inflectionPoints.size(), 1) << "x³ has inflection at x=0";
}

TEST_F(CurveFeatureDetectorTest, MandatoryAnchorsIncludeEndpoints) {
    createLinearCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    EXPECT_GE(features.mandatoryAnchors.size(), 2) << "Must always have at least endpoints";
    EXPECT_EQ(0, features.mandatoryAnchors.front()) << "First anchor should be index 0";
    EXPECT_EQ(255, features.mandatoryAnchors.back()) << "Last anchor should be index 255";
}

TEST_F(CurveFeatureDetectorTest, LinearCurveHasNoFeatures) {
    createLinearCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    EXPECT_EQ(0, features.localExtrema.size()) << "Linear curve has no extrema";
    EXPECT_EQ(0, features.inflectionPoints.size()) << "Linear curve has no inflection points";
    EXPECT_EQ(2, features.mandatoryAnchors.size()) << "Only endpoints for linear curve";
}

// ============================================================================
// Mandatory Anchors Tests
// ============================================================================

TEST_F(CurveFeatureDetectorTest, MandatoryAnchorsAreSorted) {
    createSineCurve(2);
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    for (size_t i = 1; i < features.mandatoryAnchors.size(); ++i) {
        EXPECT_LT(features.mandatoryAnchors[i-1], features.mandatoryAnchors[i])
            << "Mandatory anchors should be in ascending order";
    }
}

TEST_F(CurveFeatureDetectorTest, MandatoryAnchorsAreUnique) {
    createSineCurve(2);
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    for (size_t i = 1; i < features.mandatoryAnchors.size(); ++i) {
        EXPECT_NE(features.mandatoryAnchors[i-1], features.mandatoryAnchors[i])
            << "Mandatory anchors should not have duplicates";
    }
}

TEST_F(CurveFeatureDetectorTest, MandatoryAnchorsIncludeAllExtrema) {
    createSineCurve(1);
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // All local extrema should be in mandatory anchors
    for (int extremumIdx : features.localExtrema) {
        auto it = std::find(features.mandatoryAnchors.begin(),
                           features.mandatoryAnchors.end(),
                           extremumIdx);
        EXPECT_NE(it, features.mandatoryAnchors.end())
            << "Extremum at index " << extremumIdx << " should be in mandatory anchors";
    }
}

TEST_F(CurveFeatureDetectorTest, MandatoryAnchorsIncludeAllInflectionPoints) {
    createCubicCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // All inflection points should be in mandatory anchors
    for (int inflectionIdx : features.inflectionPoints) {
        auto it = std::find(features.mandatoryAnchors.begin(),
                           features.mandatoryAnchors.end(),
                           inflectionIdx);
        EXPECT_NE(it, features.mandatoryAnchors.end())
            << "Inflection point at index " << inflectionIdx << " should be in mandatory anchors";
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CurveFeatureDetectorTest, HandlesConstantCurve) {
    // Set all values to 0.5
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 0.5);
    }

    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    EXPECT_EQ(0, features.localExtrema.size()) << "Constant curve has no extrema";
    EXPECT_EQ(0, features.inflectionPoints.size()) << "Constant curve has no inflection points";
    EXPECT_EQ(2, features.mandatoryAnchors.size()) << "Only endpoints for constant curve";
}

TEST_F(CurveFeatureDetectorTest, HandlesStepFunction) {
    // Step function at x=0
    for (int i = 0; i < 256; ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x < 0.0 ? -0.5 : 0.5);
    }

    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // Step function has discontinuous derivative, may or may not detect extremum
    // Just verify we get reasonable output
    EXPECT_GE(features.mandatoryAnchors.size(), 2) << "At least endpoints";
    EXPECT_LE(features.mandatoryAnchors.size(), 10) << "Should not have excessive anchors";
}

// ============================================================================
// Derivative Estimation Tests
// ============================================================================

TEST_F(CurveFeatureDetectorTest, DerivativeEstimationIsReasonable) {
    createLinearCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // For linear curve, derivative should be constant (slope = 1)
    // No sign changes means no extrema
    EXPECT_EQ(0, features.localExtrema.size());
}

TEST_F(CurveFeatureDetectorTest, SecondDerivativeDetectsInflection) {
    createCubicCurve();
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf);

    // x³ has d²y/dx² = 6x, which changes sign at x=0
    EXPECT_GE(features.inflectionPoints.size(), 1);

    // Inflection should be near center (index 128)
    if (features.inflectionPoints.size() > 0) {
        int inflectionIdx = features.inflectionPoints[0];
        EXPECT_NEAR(inflectionIdx, 128, 10) << "Inflection of x³ should be near center";
    }
}

// ============================================================================
// Task 2: Significance Filtering Tests
// ============================================================================

/**
 * Test: Noisy scribble with many tiny bumps + a few major peaks
 * Expected: Filter out minor bumps, keep only significant peaks
 */
TEST_F(CurveFeatureDetectorTest, SignificanceFilter_NoisyScribble_FiltersMinorBumps) {
    // Create curve: identity + many tiny bumps + 3 major peaks
    for (int i = 0; i < 256; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = x;  // Base identity

        // Add 50 tiny bumps (±0.002 amplitude = 0.1% of vertical range)
        y += 0.002 * std::sin(50.0 * M_PI * x);

        // Add 3 major peaks (±0.3 amplitude = 15% of vertical range)
        if (x > -0.6 && x < -0.4) {
            y += 0.3 * std::sin(5.0 * M_PI * (x + 0.5));
        }
        if (x > -0.1 && x < 0.1) {
            y += 0.3 * std::sin(5.0 * M_PI * x);
        }
        if (x > 0.4 && x < 0.6) {
            y += 0.3 * std::sin(5.0 * M_PI * (x - 0.5));
        }

        ltf->setBaseLayerValue(i, y);
    }

    // Apply significance filtering: 2% threshold (should filter out 0.1% bumps but keep 15% peaks)
    dsp_core::Services::CurveFeatureDetector::FeatureDetectionConfig config;
    config.significanceThreshold = 0.02;  // 2% of vertical range
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, config);

    // Should detect only the 3 major peaks (each has 2 extrema = 6 total)
    // Plus minor noise tolerance
    EXPECT_LE(features.localExtrema.size(), 10)
        << "Should filter out minor bumps, got " << features.localExtrema.size() << " extrema";
    EXPECT_GE(features.localExtrema.size(), 4)
        << "Should detect major peaks, got only " << features.localExtrema.size() << " extrema";
}

/**
 * Test: Smooth monotonic curve (tanh) should have NO extrema detected
 * Expected: 0 extrema (monotonic curve)
 */
TEST_F(CurveFeatureDetectorTest, SignificanceFilter_SmoothCurve_NoFalsePositives) {
    createTanhCurve();

    dsp_core::Services::CurveFeatureDetector::FeatureDetectionConfig config;
    config.significanceThreshold = 0.01;  // Even with low threshold
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, config);

    EXPECT_EQ(0, features.localExtrema.size())
        << "Smooth monotonic tanh should have no extrema";
}

/**
 * Test: Sine wave with full amplitude should preserve all extrema
 * Expected: All significant extrema detected
 */
TEST_F(CurveFeatureDetectorTest, SignificanceFilter_LargeExtrema_AllPreserved) {
    createSineCurve(2);  // 2 periods = 4 extrema

    // Apply very low significance threshold to ensure all extrema are detected
    // Note: With 256 samples over 2 periods, local amplitude at peaks is relatively small
    dsp_core::Services::CurveFeatureDetector::FeatureDetectionConfig config;
    config.significanceThreshold = 0.001;  // 0.1% of range (very permissive)
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, config);

    // Sine wave has full amplitude (2.0 range), so all extrema should be preserved
    // 2 periods = 2 peaks + 2 valleys = 4 extrema
    EXPECT_GE(features.localExtrema.size(), 3)
        << "Should detect all major extrema in sine wave";
    EXPECT_LE(features.localExtrema.size(), 5)
        << "Should not detect spurious extrema";
}

/**
 * Test: Backward compatibility - legacy overload still works
 */
TEST_F(CurveFeatureDetectorTest, LegacyOverload_WorksCorrectly) {
    createSineCurve(1);

    // Use legacy overload (maxMandatoryAnchors parameter)
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, 5);

    EXPECT_LE(features.mandatoryAnchors.size(), static_cast<size_t>(5))
        << "Legacy overload should respect maxMandatoryAnchors limit";
    EXPECT_GE(features.mandatoryAnchors.size(), 2)
        << "Should always have at least endpoints";
}

} // namespace dsp_core_test
