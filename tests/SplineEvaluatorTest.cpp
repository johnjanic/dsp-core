#include <gtest/gtest.h>
#include "dsp_core/Source/Services/SplineEvaluator.h"
#include <cmath>

using namespace dsp_core;
using namespace dsp_core::Services;

//==============================================================================
// Test Fixtures
//==============================================================================

class SplineEvaluatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create simple test anchors
        linearAnchors = {{-1.0, -1.0, false, 1.0}, // y = x, slope = 1
                         {1.0, 1.0, false, 1.0}};

        // Three-point S-curve
        sCurveAnchors = {
            {-1.0, -1.0, false, 0.0}, // Start flat
            {0.0, 0.0, false, 2.0},   // Steep middle
            {1.0, 1.0, false, 0.0}    // End flat
        };

        // Monotonic increasing curve
        monotonicAnchors = {{-1.0, -1.0, false, 0.5},
                            {-0.5, -0.7, false, 0.8},
                            {0.0, 0.0, false, 1.0},
                            {0.5, 0.7, false, 0.8},
                            {1.0, 1.0, false, 0.5}};
    }

    std::vector<SplineAnchor> linearAnchors;
    std::vector<SplineAnchor> sCurveAnchors;
    std::vector<SplineAnchor> monotonicAnchors;

    static constexpr double kTolerance = 1e-9;
};

//==============================================================================
// Basic Evaluation Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, EmptyAnchorsReturnsZero) {
    std::vector<SplineAnchor> empty;
    EXPECT_DOUBLE_EQ(SplineEvaluator::evaluate(empty, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(SplineEvaluator::evaluate(empty, 0.5), 0.0);
}

TEST_F(SplineEvaluatorTest, SingleAnchorReturnsConstant) {
    std::vector<SplineAnchor> single = {{0.5, 0.3, false, 0.0}};
    EXPECT_DOUBLE_EQ(SplineEvaluator::evaluate(single, -1.0), 0.3);
    EXPECT_DOUBLE_EQ(SplineEvaluator::evaluate(single, 0.0), 0.3);
    EXPECT_DOUBLE_EQ(SplineEvaluator::evaluate(single, 1.0), 0.3);
}

TEST_F(SplineEvaluatorTest, EvaluateAtAnchorPositionsReturnsExactValues) {
    // Linear case
    EXPECT_NEAR(SplineEvaluator::evaluate(linearAnchors, -1.0), -1.0, kTolerance);
    EXPECT_NEAR(SplineEvaluator::evaluate(linearAnchors, 1.0), 1.0, kTolerance);

    // S-curve case
    EXPECT_NEAR(SplineEvaluator::evaluate(sCurveAnchors, -1.0), -1.0, kTolerance);
    EXPECT_NEAR(SplineEvaluator::evaluate(sCurveAnchors, 0.0), 0.0, kTolerance);
    EXPECT_NEAR(SplineEvaluator::evaluate(sCurveAnchors, 1.0), 1.0, kTolerance);

    // Monotonic case
    for (const auto& anchor : monotonicAnchors) {
        double result = SplineEvaluator::evaluate(monotonicAnchors, anchor.x);
        EXPECT_NEAR(result, anchor.y, kTolerance) << "Failed at anchor x=" << anchor.x;
    }
}

TEST_F(SplineEvaluatorTest, LinearInterpolationWithUniformSlope) {
    // For y = x with slope = 1 everywhere, should be exact linear interpolation
    double x = 0.0;
    double y = SplineEvaluator::evaluate(linearAnchors, x);
    EXPECT_NEAR(y, 0.0, kTolerance);

    x = 0.5;
    y = SplineEvaluator::evaluate(linearAnchors, x);
    EXPECT_NEAR(y, 0.5, kTolerance);

    x = -0.5;
    y = SplineEvaluator::evaluate(linearAnchors, x);
    EXPECT_NEAR(y, -0.5, kTolerance);
}

//==============================================================================
// Boundary Condition Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, EvaluateBeforeFirstAnchorReturnsFirstValue) {
    double result = SplineEvaluator::evaluate(linearAnchors, -2.0);
    EXPECT_DOUBLE_EQ(result, linearAnchors.front().y);

    result = SplineEvaluator::evaluate(sCurveAnchors, -10.0);
    EXPECT_DOUBLE_EQ(result, sCurveAnchors.front().y);
}

TEST_F(SplineEvaluatorTest, EvaluateAfterLastAnchorReturnsLastValue) {
    double result = SplineEvaluator::evaluate(linearAnchors, 2.0);
    EXPECT_DOUBLE_EQ(result, linearAnchors.back().y);

    result = SplineEvaluator::evaluate(sCurveAnchors, 10.0);
    EXPECT_DOUBLE_EQ(result, sCurveAnchors.back().y);
}

//==============================================================================
// Hermite Polynomial Correctness Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, HermiteInterpolationPreservesEndpointTangents) {
    // At segment start (t=0), derivative should match p0.tangent
    // At segment end (t=1), derivative should match p1.tangent

    // For linear case with slope=1, derivative should be 1 everywhere
    double deriv_start = SplineEvaluator::evaluateDerivative(linearAnchors, -1.0);
    EXPECT_NEAR(deriv_start, 1.0, 1e-6);

    double deriv_end = SplineEvaluator::evaluateDerivative(linearAnchors, 1.0);
    EXPECT_NEAR(deriv_end, 1.0, 1e-6);
}

TEST_F(SplineEvaluatorTest, SCurveHasFlatEndsAndSteepMiddle) {
    // S-curve should have near-zero derivatives at endpoints
    double deriv_start = SplineEvaluator::evaluateDerivative(sCurveAnchors, -1.0);
    EXPECT_NEAR(deriv_start, 0.0, 1e-6);

    double deriv_end = SplineEvaluator::evaluateDerivative(sCurveAnchors, 1.0);
    EXPECT_NEAR(deriv_end, 0.0, 1e-6);

    // Middle should have steeper slope
    double deriv_middle = SplineEvaluator::evaluateDerivative(sCurveAnchors, 0.0);
    EXPECT_GT(deriv_middle, 1.5); // Should be close to 2.0
}

//==============================================================================
// Monotonicity Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, MonotonicAnchorsProduceMonotonicCurve) {
    // Sample curve at many points and verify monotonicity
    constexpr int numSamples = 100;
    double prevY = SplineEvaluator::evaluate(monotonicAnchors, -1.0);

    for (int i = 1; i <= numSamples; ++i) {
        double x = -1.0 + (2.0 * i / numSamples);
        double y = SplineEvaluator::evaluate(monotonicAnchors, x);

        EXPECT_GE(y, prevY - kTolerance) << "Monotonicity violated at x=" << x << " (y=" << y << ", prevY=" << prevY
                                         << ")";

        prevY = y;
    }
}

//==============================================================================
// Edge Case Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, DegenerateSegmentWithZeroDx) {
    // Two anchors at same x position (degenerate)
    std::vector<SplineAnchor> degenerate = {
        {0.0, -1.0, false, 0.0}, {0.0, 1.0, false, 0.0} // Same x as previous
    };

    // Should return first anchor's y value
    double result = SplineEvaluator::evaluate(degenerate, 0.0);
    EXPECT_DOUBLE_EQ(result, -1.0);
}

TEST_F(SplineEvaluatorTest, VeryCloseAnchors) {
    // Two anchors very close together
    std::vector<SplineAnchor> close = {{-1.0, -1.0, false, 1.0}, {-0.999999, -0.999999, false, 1.0}};

    // Should still interpolate correctly
    double result = SplineEvaluator::evaluate(close, -0.9999995);
    EXPECT_NEAR(result, -0.9999995, 1e-5);
}

//==============================================================================
// Derivative Evaluation Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, DerivativeOfConstantIsZero) {
    std::vector<SplineAnchor> flat = {{-1.0, 0.5, false, 0.0}, {1.0, 0.5, false, 0.0}};

    // Derivative should be zero everywhere for flat curve
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(flat, -0.5), 0.0, 1e-6);
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(flat, 0.0), 0.0, 1e-6);
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(flat, 0.5), 0.0, 1e-6);
}

TEST_F(SplineEvaluatorTest, DerivativeOfLinearIsConstant) {
    // y = x with slope = 1
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(linearAnchors, -0.5), 1.0, 1e-6);
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(linearAnchors, 0.0), 1.0, 1e-6);
    EXPECT_NEAR(SplineEvaluator::evaluateDerivative(linearAnchors, 0.5), 1.0, 1e-6);
}

TEST_F(SplineEvaluatorTest, DerivativeBeforeFirstAnchor) {
    double deriv = SplineEvaluator::evaluateDerivative(linearAnchors, -2.0);
    EXPECT_DOUBLE_EQ(deriv, linearAnchors.front().tangent);
}

TEST_F(SplineEvaluatorTest, DerivativeAfterLastAnchor) {
    double deriv = SplineEvaluator::evaluateDerivative(linearAnchors, 2.0);
    EXPECT_DOUBLE_EQ(deriv, linearAnchors.back().tangent);
}

//==============================================================================
// Numerical Derivative Comparison
//==============================================================================

TEST_F(SplineEvaluatorTest, AnalyticalDerivativeMatchesNumerical) {
    // Verify analytical derivative matches numerical approximation
    constexpr double h = 1e-6;

    for (double x = -0.9; x <= 0.9; x += 0.2) {
        double f_plus = SplineEvaluator::evaluate(monotonicAnchors, x + h);
        double f_minus = SplineEvaluator::evaluate(monotonicAnchors, x - h);
        double numerical_deriv = (f_plus - f_minus) / (2.0 * h);

        double analytical_deriv = SplineEvaluator::evaluateDerivative(monotonicAnchors, x);

        EXPECT_NEAR(analytical_deriv, numerical_deriv, 1e-3) << "Derivative mismatch at x=" << x;
    }
}

//==============================================================================
// Integration Tests with Known Analytical Curves
//==============================================================================

TEST_F(SplineEvaluatorTest, QuadraticCurveApproximation) {
    // Fit y = x² using PCHIP anchors
    // Expected tangents at sample points
    std::vector<SplineAnchor> quadratic = {
        {-1.0, 1.0, false, -2.0},  // y' = 2x at x=-1
        {-0.5, 0.25, false, -1.0}, // y' = 2x at x=-0.5
        {0.0, 0.0, false, 0.0},    // y' = 2x at x=0
        {0.5, 0.25, false, 1.0},   // y' = 2x at x=0.5
        {1.0, 1.0, false, 2.0}     // y' = 2x at x=1
    };

    // Sample and verify approximation quality
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        double expected = x * x;
        double actual = SplineEvaluator::evaluate(quadratic, x);

        // PCHIP should approximate x² very well
        EXPECT_NEAR(actual, expected, 0.01) << "Quadratic approximation poor at x=" << x;
    }
}

//==============================================================================
// Custom Tangent Override Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, CustomTangentOverride) {
    std::vector<SplineAnchor> custom = {
        {-1.0, -1.0, true, 0.0}, // Override to flat start
        {0.0, 0.0, false, 1.0},  // Use computed tangent
        {1.0, 1.0, true, 0.0}    // Override to flat end
    };

    // Start and end should be flat
    double deriv_start = SplineEvaluator::evaluateDerivative(custom, -1.0);
    EXPECT_NEAR(deriv_start, 0.0, 1e-6);

    double deriv_end = SplineEvaluator::evaluateDerivative(custom, 1.0);
    EXPECT_NEAR(deriv_end, 0.0, 1e-6);
}

//==============================================================================
// Batch Evaluation Tests
//==============================================================================

TEST_F(SplineEvaluatorTest, BatchEvaluateMatchesSingleEvaluate) {
    // Test that batch evaluation produces identical results to individual calls
    const int count = 256;
    std::vector<double> xValues(count);
    std::vector<double> yBatch(count);

    // Generate sorted X values from -1 to 1
    for (int i = 0; i < count; ++i) {
        xValues[i] = -1.0 + (2.0 * i / (count - 1));
    }

    // Batch evaluate
    SplineEvaluator::evaluateBatch(monotonicAnchors, xValues.data(), yBatch.data(), count);

    // Compare with individual evaluations
    for (int i = 0; i < count; ++i) {
        double yIndividual = SplineEvaluator::evaluate(monotonicAnchors, xValues[i]);
        EXPECT_NEAR(yBatch[i], yIndividual, kTolerance) << "Mismatch at index " << i << ", x=" << xValues[i];
    }
}

TEST_F(SplineEvaluatorTest, BatchEvaluateHandlesEmptyAnchors) {
    const int count = 10;
    std::vector<double> xValues(count, 0.5);
    std::vector<double> yValues(count);

    std::vector<SplineAnchor> empty;
    SplineEvaluator::evaluateBatch(empty, xValues.data(), yValues.data(), count);

    for (int i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(yValues[i], 0.0);
    }
}

TEST_F(SplineEvaluatorTest, BatchEvaluateHandlesSingleAnchor) {
    const int count = 10;
    std::vector<double> xValues(count);
    std::vector<double> yValues(count);

    for (int i = 0; i < count; ++i) {
        xValues[i] = -1.0 + (2.0 * i / (count - 1));
    }

    std::vector<SplineAnchor> single = {{0.5, 0.7, false, 0.0}};
    SplineEvaluator::evaluateBatch(single, xValues.data(), yValues.data(), count);

    for (int i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(yValues[i], 0.7);
    }
}

TEST_F(SplineEvaluatorTest, BatchEvaluateHandlesBoundaryConditions) {
    const int count = 20;
    std::vector<double> xValues(count);
    std::vector<double> yValues(count);

    // Test values before, within, and after anchor range
    for (int i = 0; i < count; ++i) {
        xValues[i] = -2.0 + (4.0 * i / (count - 1)); // Range [-2, 2]
    }

    SplineEvaluator::evaluateBatch(sCurveAnchors, // Anchors range from -1 to 1
                                   xValues.data(), yValues.data(), count);

    // Values before first anchor should equal first anchor's y
    for (int i = 0; i < count; ++i) {
        if (xValues[i] < -1.0) {
            EXPECT_DOUBLE_EQ(yValues[i], -1.0) << "Before range at x=" << xValues[i];
        } else if (xValues[i] > 1.0) {
            EXPECT_DOUBLE_EQ(yValues[i], 1.0) << "After range at x=" << xValues[i];
        }
    }
}
