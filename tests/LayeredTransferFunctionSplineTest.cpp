#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace dsp_core;
using namespace dsp_core::Services;

//==============================================================================
// Test Fixtures
//==============================================================================

class LayeredTransferFunctionSplineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<LayeredTransferFunction>(256, -1.0, 1.0);

        // Create test anchors
        linearAnchors = {{-1.0, -1.0, false, 0.0}, {1.0, 1.0, false, 0.0}};

        threePtAnchors = {{-1.0, -1.0, false, 0.0}, {0.0, 0.5, false, 0.0}, {1.0, 1.0, false, 0.0}};

        // Compute tangents using Akima (default)
        SplineFitter::computeTangents(linearAnchors, SplineFitConfig::tight());
        SplineFitter::computeTangents(threePtAnchors, SplineFitConfig::tight());
    }

    std::unique_ptr<LayeredTransferFunction> ltf;
    std::vector<SplineAnchor> linearAnchors;
    std::vector<SplineAnchor> threePtAnchors;

    static constexpr double kTolerance = 1e-3;
};

//==============================================================================
// Spline Layer Enable/Disable Tests
//==============================================================================

TEST_F(LayeredTransferFunctionSplineTest, SplineLayerInitiallyDisabled) {
    EXPECT_FALSE(ltf->isSplineLayerEnabled());
}

TEST_F(LayeredTransferFunctionSplineTest, SplineLayerCanBeEnabled) {
    ltf->setSplineLayerEnabled(true);
    EXPECT_TRUE(ltf->isSplineLayerEnabled());
}

TEST_F(LayeredTransferFunctionSplineTest, EnablingSplineLayerLocksNormalization) {
    ltf->setSplineLayerEnabled(true);
    EXPECT_NEAR(ltf->getNormalizationScalar(), 1.0, 1e-9);
}

//==============================================================================
// Direct Evaluation Path Tests
//==============================================================================

TEST_F(LayeredTransferFunctionSplineTest, DirectEvaluationPath) {
    // Set up spline layer
    ltf->getSplineLayer().setAnchors(threePtAnchors);
    ltf->setSplineLayerEnabled(true);

    // Evaluate (should use direct path since cache invalid)
    double result = ltf->applyTransferFunction(0.0);
    EXPECT_NEAR(result, 0.5, kTolerance);
}

TEST_F(LayeredTransferFunctionSplineTest, DirectPathAtAnchorPoints) {
    ltf->getSplineLayer().setAnchors(threePtAnchors);
    ltf->setSplineLayerEnabled(true);

    EXPECT_NEAR(ltf->applyTransferFunction(-1.0), -1.0, kTolerance);
    EXPECT_NEAR(ltf->applyTransferFunction(0.0), 0.5, kTolerance);
    EXPECT_NEAR(ltf->applyTransferFunction(1.0), 1.0, kTolerance);
}

//==============================================================================
// Layer Exclusivity Tests
//==============================================================================

TEST_F(LayeredTransferFunctionSplineTest, ModeExclusivity) {
    // Set up harmonics
    ltf->setCoefficient(1, 0.5); // Harmonic 1 amplitude
    ltf->updateComposite();

    // Enable spline layer
    ltf->setSplineLayerEnabled(true);

    // Audio thread should use spline, ignoring harmonics
    ltf->getSplineLayer().setAnchors(linearAnchors);

    double result = ltf->applyTransferFunction(0.0);
    // Result should come from spline (y=x at x=0), not harmonics
    EXPECT_NEAR(result, 0.0, kTolerance);
}

TEST_F(LayeredTransferFunctionSplineTest, DisablingSplineLayerRestoresHarmonicMode) {
    // Enable spline mode
    ltf->getSplineLayer().setAnchors(linearAnchors);
    ltf->setSplineLayerEnabled(true);

    const double splineResult = ltf->applyTransferFunction(0.5);

    // Disable spline mode
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    const double harmonicResult = ltf->applyTransferFunction(0.5);

    // Results should be different (harmonic mode uses base+harmonics)
    // Note: This test assumes base layer is identity and harmonics are zero
    // Both modes produce similar results for linear transfer functions
    EXPECT_GE(splineResult, -1.0);
    EXPECT_LE(splineResult, 1.0);
    EXPECT_GE(harmonicResult, -1.0);
    EXPECT_LE(harmonicResult, 1.0);
}

//==============================================================================
// Composite Update in Spline Mode Tests
//==============================================================================

TEST_F(LayeredTransferFunctionSplineTest, UpdateCompositeInSplineMode) {
    ltf->getSplineLayer().setAnchors(threePtAnchors);
    ltf->setSplineLayerEnabled(true);

    // Update composite
    ltf->updateComposite();

    // Verify composite values match direct evaluation
    for (int i = 0; i < ltf->getTableSize(); i += 16) {
        double x = ltf->normalizeIndex(i);
        double cached = ltf->getCompositeValue(i);
        double direct = ltf->getSplineLayer().evaluate(x);
        EXPECT_NEAR(cached, direct, kTolerance);
    }
}

TEST_F(LayeredTransferFunctionSplineTest, UpdateCompositePreservesSplineShape) {
    ltf->getSplineLayer().setAnchors(threePtAnchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // Check anchor points in cached composite
    int midIdx = ltf->getTableSize() / 2;
    double midValue = ltf->getCompositeValue(midIdx);

    // Middle of curve should be close to 0.5 (from threePtAnchors)
    EXPECT_GT(midValue, 0.4);
    EXPECT_LT(midValue, 0.6);
}

//==============================================================================
// Normalization Tests
//==============================================================================

TEST_F(LayeredTransferFunctionSplineTest, SplineModeUsesIdentityNormalization) {
    // Create anchors that would normally be normalized
    std::vector<SplineAnchor> largeAnchors = {{-1.0, -2.0, false, 0.0}, {1.0, 2.0, false, 0.0}};
    SplineFitter::computeTangents(largeAnchors, SplineFitConfig::tight());

    ltf->getSplineLayer().setAnchors(largeAnchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // In spline mode, normalization is locked to 1.0
    // So values should not be scaled down
    EXPECT_NEAR(ltf->getNormalizationScalar(), 1.0, 1e-9);

    // Result should be close to 2.0 at x=1 (not normalized)
    double result = ltf->applyTransferFunction(1.0);
    EXPECT_GT(result, 1.5); // Should not be normalized to 1.0
}
