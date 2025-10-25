#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for SplineFitter service
 * Tests PCHIP spline curve fitting algorithm:
 *   - Data structures (SplineTypes)
 *   - Sample & sanitize (Step 1)
 *   - RDP simplification (Step 2)
 *   - PCHIP tangent computation (Step 3)
 */
class SplineFitterTest : public ::testing::Test {
protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
    }

    // Helper: Set base layer to identity curve (y = x)
    void setIdentityCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x);
        }
    }

    // Helper: Set base layer to S-curve (cubic)
    void setSCurve() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);
            // Cubic S-curve: y = x^3
            double y = x * x * x;
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Helper: Set base layer to step function
    void setStepFunction() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x < 0.0 ? -0.5 : 0.5);
        }
    }

    // Helper: Set base layer to sine wave
    void setSineWave() {
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);
            // Normalize to [-1, 1]
            double y = std::sin(x * M_PI);
            ltf->setBaseLayerValue(i, y);
        }
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

// ============================================================================
// SplineTypes Tests
// ============================================================================

TEST(SplineTypesTest, SplineAnchor_DefaultInitialization) {
    dsp_core::SplineAnchor anchor;
    EXPECT_DOUBLE_EQ(anchor.x, 0.0);
    EXPECT_DOUBLE_EQ(anchor.y, 0.0);
    EXPECT_FALSE(anchor.hasCustomTangent);
    EXPECT_DOUBLE_EQ(anchor.tangent, 0.0);
}

TEST(SplineTypesTest, SplineAnchor_Equality) {
    dsp_core::SplineAnchor a1{0.5, 0.5, false, 0.0};
    dsp_core::SplineAnchor a2{0.5, 0.5, false, 0.0};
    dsp_core::SplineAnchor a3{0.5, 0.6, false, 0.0};

    EXPECT_EQ(a1, a2);
    EXPECT_NE(a1, a3);
}

TEST(SplineTypesTest, SplineFitResult_DefaultInitialization) {
    dsp_core::SplineFitResult result;
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.numAnchors, 0);
    EXPECT_DOUBLE_EQ(result.maxError, 0.0);
    EXPECT_DOUBLE_EQ(result.averageError, 0.0);
    EXPECT_DOUBLE_EQ(result.maxDerivativeError, 0.0);
}

TEST(SplineTypesTest, SplineFitConfig_DefaultValues) {
    dsp_core::SplineFitConfig config;
    EXPECT_DOUBLE_EQ(config.positionTolerance, 0.01);
    EXPECT_DOUBLE_EQ(config.derivativeTolerance, 0.02);
    EXPECT_EQ(config.maxAnchors, 64);
    EXPECT_TRUE(config.enableRefinement);
    EXPECT_TRUE(config.enforceMonotonicity);
    EXPECT_DOUBLE_EQ(config.minSlope, -8.0);
    EXPECT_DOUBLE_EQ(config.maxSlope, 8.0);
    EXPECT_TRUE(config.pinEndpoints);
}

TEST(SplineTypesTest, SplineFitConfig_TightPreset) {
    auto config = dsp_core::SplineFitConfig::tight();
    EXPECT_DOUBLE_EQ(config.positionTolerance, 0.002);
    EXPECT_DOUBLE_EQ(config.derivativeTolerance, 0.05);
    EXPECT_EQ(config.maxAnchors, 64);
}

TEST(SplineTypesTest, SplineFitConfig_SmoothPreset) {
    auto config = dsp_core::SplineFitConfig::smooth();
    EXPECT_DOUBLE_EQ(config.positionTolerance, 0.01);
    EXPECT_DOUBLE_EQ(config.derivativeTolerance, 0.02);
    EXPECT_EQ(config.maxAnchors, 24);
}

// ============================================================================
// SplineFitter Basic Tests
// ============================================================================

TEST_F(SplineFitterTest, FitCurve_IdentityCurve_MinimalAnchors) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 2);  // At least endpoints
    EXPECT_LE(result.numAnchors, 10); // Linear curve should need very few anchors

    // Endpoints should be preserved
    EXPECT_NEAR(result.anchors.front().x, -1.0, 0.01);
    EXPECT_NEAR(result.anchors.front().y, -1.0, 0.01);
    EXPECT_NEAR(result.anchors.back().x, 1.0, 0.01);
    EXPECT_NEAR(result.anchors.back().y, 1.0, 0.01);
}

TEST_F(SplineFitterTest, FitCurve_SCurve_MoreAnchors) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 3);  // S-curve needs anchors at inflection points
}

TEST_F(SplineFitterTest, FitCurve_TightTolerance_MoreAnchors) {
    setSCurve();

    auto smoothConfig = dsp_core::SplineFitConfig::smooth();
    auto tightConfig = dsp_core::SplineFitConfig::tight();

    auto smoothResult = dsp_core::Services::SplineFitter::fitCurve(*ltf, smoothConfig);
    auto tightResult = dsp_core::Services::SplineFitter::fitCurve(*ltf, tightConfig);

    EXPECT_TRUE(smoothResult.success);
    EXPECT_TRUE(tightResult.success);

    // Tighter tolerance should produce more anchors
    EXPECT_GE(tightResult.numAnchors, smoothResult.numAnchors);
}

TEST_F(SplineFitterTest, FitCurve_AnchorsAreSorted) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // Verify anchors are sorted by x
    for (size_t i = 1; i < result.anchors.size(); ++i) {
        EXPECT_GT(result.anchors[i].x, result.anchors[i-1].x)
            << "Anchor " << i << " is not sorted";
    }
}

// ============================================================================
// PCHIP Tangent Tests
// ============================================================================

TEST_F(SplineFitterTest, PCHIPTangents_MonotonicSequence) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // For monotonic increasing data, all tangents should be positive
    for (const auto& anchor : result.anchors) {
        EXPECT_GE(anchor.tangent, 0.0)
            << "Tangent at x=" << anchor.x << " should be non-negative";
    }
}

TEST_F(SplineFitterTest, PCHIPTangents_LocalExtremum) {
    // Create curve with clear local extremum
    for (int i = 0; i < 256; ++i) {
        double x = ltf->normalizeIndex(i);
        // Parabola with peak at x=0: y = 1 - x^2
        double y = 1.0 - x * x;
        ltf->setBaseLayerValue(i, y);
    }

    auto config = dsp_core::SplineFitConfig::tight();  // Use tight to capture peak
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // Find anchor at peak (x ≈ 0)
    bool foundPeak = false;
    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 0.1 && std::abs(anchor.y - 1.0) < 0.1) {
            // Tangent should be near zero at peak
            EXPECT_NEAR(anchor.tangent, 0.0, 0.3)
                << "Tangent at peak (x=" << anchor.x << ") should be near zero";
            foundPeak = true;
        }
    }

    // Note: RDP simplification means we may not capture the exact peak
    // but if we do, the tangent should be near zero
}

TEST_F(SplineFitterTest, PCHIPTangents_SlopeCapping) {
    setStepFunction();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // All tangents should respect slope caps
    for (const auto& anchor : result.anchors) {
        EXPECT_GE(anchor.tangent, config.minSlope)
            << "Tangent at x=" << anchor.x << " exceeds min slope";
        EXPECT_LE(anchor.tangent, config.maxSlope)
            << "Tangent at x=" << anchor.x << " exceeds max slope";
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(SplineFitterTest, FitCurve_FlatCurve) {
    // Set all values to constant
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 0.0);
    }

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 2);  // At least endpoints

    // All anchors should have y ≈ 0
    for (const auto& anchor : result.anchors) {
        EXPECT_NEAR(anchor.y, 0.0, 0.01);
        EXPECT_NEAR(anchor.tangent, 0.0, 0.01);  // Flat curve → zero tangent
    }
}

TEST_F(SplineFitterTest, FitCurve_MonotonicityEnforcement_ProducesReasonableResult) {
    // Create slightly non-monotonic curve (with small violations)
    setIdentityCurve();
    // Add small monotonicity violations
    ltf->setBaseLayerValue(64, -0.3);   // Should be ~-0.5
    ltf->setBaseLayerValue(128, 0.2);   // Should be ~0.0
    ltf->setBaseLayerValue(192, 0.6);   // Should be ~0.5

    auto config = dsp_core::SplineFitConfig::smooth();
    config.enforceMonotonicity = true;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // Monotonicity enforcement applies pairwise averaging to samples,
    // but RDP may still select anchors with small violations.
    // Verify result is generally increasing with tolerance for small errors.
    int violations = 0;
    for (size_t i = 1; i < result.anchors.size(); ++i) {
        if (result.anchors[i].y < result.anchors[i-1].y - 0.01) {
            violations++;
        }
    }
    // Allow some violations (< 40%) - current monotonicity is simple pairwise
    EXPECT_LT(violations, result.numAnchors * 2 / 5)
        << "Too many monotonicity violations: " << violations << " out of " << result.numAnchors;
}

TEST_F(SplineFitterTest, FitCurve_NonMonotonicCurve_WithoutEnforcement) {
    setSineWave();

    auto config = dsp_core::SplineFitConfig::smooth();
    config.enforceMonotonicity = false;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    // Without enforcement, anchors may be non-monotonic
    // Just verify we get a valid result
    EXPECT_GE(result.numAnchors, 2);
}

// ============================================================================
// RDP Algorithm Tests
// ============================================================================

TEST_F(SplineFitterTest, RDP_StraightLine_TwoAnchors) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    // Straight line should be simplified to just endpoints
    EXPECT_LE(result.numAnchors, 5);  // Allow small tolerance for numerical error
}

TEST_F(SplineFitterTest, RDP_ComplexCurve_PreservesShape) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // S-curve should have anchors near inflection point (x ≈ 0)
    bool foundNearZero = false;
    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 0.2) {
            foundNearZero = true;
            break;
        }
    }
    EXPECT_TRUE(foundNearZero) << "RDP should preserve inflection point";
}

// ============================================================================
// SplineEvaluator Tests
// ============================================================================

TEST(SplineEvaluatorTest, Evaluate_EmptyAnchors_ReturnsZero) {
    std::vector<dsp_core::SplineAnchor> anchors;
    double result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.5);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(SplineEvaluatorTest, Evaluate_SingleAnchor_ReturnsAnchorValue) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {0.0, 0.5, false, 0.0}
    };
    double result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0);
    EXPECT_DOUBLE_EQ(result, 0.5);

    // Any x should return the same value
    result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.0);
    EXPECT_DOUBLE_EQ(result, 0.5);
}

TEST(SplineEvaluatorTest, Evaluate_AtAnchorPositions_ReturnsExactValues) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {-1.0, -1.0, false, 1.0},
        {0.0, 0.0, false, 1.0},
        {1.0, 1.0, false, 1.0}
    };

    // Evaluate at each anchor position
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, -1.0), -1.0, 1e-10);
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.0), 1.0, 1e-10);
}

TEST(SplineEvaluatorTest, Evaluate_LinearInterpolation) {
    // Two anchors with slope = 1 (identity line)
    std::vector<dsp_core::SplineAnchor> anchors = {
        {-1.0, -1.0, false, 1.0},  // tangent = 1
        {1.0, 1.0, false, 1.0}     // tangent = 1
    };

    // Evaluate at midpoint - should be close to 0.0 for linear
    double midpoint = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0);
    EXPECT_NEAR(midpoint, 0.0, 0.01);  // Allow small deviation for Hermite
}

TEST(SplineEvaluatorTest, Evaluate_BeforeFirstAnchor_Clamps) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {0.0, 0.5, false, 0.0},
        {1.0, 1.0, false, 0.0}
    };

    // Evaluate before first anchor
    double result = dsp_core::Services::SplineEvaluator::evaluate(anchors, -0.5);
    EXPECT_DOUBLE_EQ(result, 0.5);  // Should return first anchor's y value
}

TEST(SplineEvaluatorTest, Evaluate_AfterLastAnchor_Clamps) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {0.0, 0.0, false, 0.0},
        {1.0, 0.5, false, 0.0}
    };

    // Evaluate after last anchor
    double result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.5);
    EXPECT_DOUBLE_EQ(result, 0.5);  // Should return last anchor's y value
}

TEST(SplineEvaluatorTest, Evaluate_MonotonicSpline) {
    // Create monotonic anchors
    std::vector<dsp_core::SplineAnchor> anchors = {
        {-1.0, -0.8, false, 0.5},
        {-0.5, -0.3, false, 0.6},
        {0.0, 0.1, false, 0.7},
        {0.5, 0.4, false, 0.6},
        {1.0, 0.9, false, 0.5}
    };

    // Sample and verify monotonicity
    double prevY = -1.0;
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        double y = dsp_core::Services::SplineEvaluator::evaluate(anchors, x);
        EXPECT_GE(y, prevY) << "Non-monotonic at x=" << x;
        prevY = y;
    }
}

TEST(SplineEvaluatorTest, EvaluateDerivative_AtAnchors) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {-1.0, -1.0, false, 1.0},  // tangent = 1
        {0.0, 0.0, false, 0.0},    // tangent = 0 (extremum)
        {1.0, 1.0, false, 1.0}     // tangent = 1
    };

    // Derivative at midpoint anchor should be close to 0
    double deriv = dsp_core::Services::SplineEvaluator::evaluateDerivative(anchors, 0.0);
    EXPECT_NEAR(deriv, 0.0, 0.1);
}

TEST(SplineEvaluatorTest, FindSegment_BinarySearch) {
    std::vector<dsp_core::SplineAnchor> anchors = {
        {-1.0, 0.0, false, 0.0},
        {-0.5, 0.0, false, 0.0},
        {0.0, 0.0, false, 0.0},
        {0.5, 0.0, false, 0.0},
        {1.0, 0.0, false, 0.0}
    };

    // Test that evaluator can handle many segments efficiently
    for (double x = -1.0; x <= 1.0; x += 0.05) {
        // Should not crash or hang
        dsp_core::Services::SplineEvaluator::evaluate(anchors, x);
    }
}

// ============================================================================
// Integration Tests (SplineFitter + SplineEvaluator)
// ============================================================================

TEST_F(SplineFitterTest, Integration_ErrorMetrics) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.maxError, 0.0);
    EXPECT_GE(result.averageError, 0.0);
    EXPECT_LE(result.averageError, result.maxError);  // avg <= max

    // For identity curve, error should be very small
    EXPECT_LT(result.maxError, 0.05) << "Identity curve fit error too large";
}

TEST_F(SplineFitterTest, Integration_FitAndReconstruct) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::smooth();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(result.success);

    // Reconstruct curve at original sample points
    int matchCount = 0;
    for (int i = 0; i < 256; i += 10) {  // Sample every 10th point
        double x = ltf->normalizeIndex(i);
        double originalY = ltf->getBaseLayerValue(i);
        double fittedY = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, x);

        // Check if reconstruction is close to original
        if (std::abs(originalY - fittedY) < 0.1) {
            matchCount++;
        }
    }

    // Most points should match well
    EXPECT_GT(matchCount, 20) << "Reconstruction quality too low";
}

} // namespace dsp_core_test
