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
        ltf->updateComposite();  // CRITICAL: Update composite after changing base layer
    }

    // Helper: Create sine curve (multiple extrema)
    void createSineCurve(int numPeriods = 2) {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            double y = std::sin(numPeriods * M_PI * x);
            ltf->setBaseLayerValue(i, y);
        }
        ltf->updateComposite();  // CRITICAL: Update composite after changing base layer
    }

    // Helper: Create cubic curve (monotonic with inflection at x=0)
    void createCubicCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            double y = x * x * x;
            ltf->setBaseLayerValue(i, y);
        }
        ltf->updateComposite();  // CRITICAL: Update composite after changing base layer
    }

    // Helper: Create linear curve (no features)
    void createLinearCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);  // [-1, 1]
            ltf->setBaseLayerValue(i, x);
        }
        ltf->updateComposite();  // CRITICAL: Update composite after changing base layer
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

} // namespace dsp_core_test
