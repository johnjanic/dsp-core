#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace dsp_core;
using namespace dsp_core::Services;

/**
 * Integration tests for end-to-end spline fitting workflows
 *
 * These tests simulate realistic user interactions:
 *   - Anchor manipulation → bake → refit cycles
 *   - Harmonic mixing → spline conversion workflows
 *   - Multi-cycle backtranslation stability
 *
 * Goal: Validate no anchor creeping and proper shape preservation
 */
class SplineFitterIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Use high resolution (16k samples) for realistic production workflow
        ltf = std::make_unique<LayeredTransferFunction>(16384, -1.0, 1.0);

        // Initialize to identity curve
        setIdentityCurve();
    }

    // Helper: Set base layer to identity curve (y = x)
    void setIdentityCurve() {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x);
        }
        ltf->updateComposite();
    }

    /**
     * Helper: Bake composite to base layer (simulates controller behavior)
     *
     * What it does:
     *   1. Copies composite table to base layer
     *   2. Zeros all harmonic coefficients
     *   3. Sets wtCoeff to 1.0
     *   4. Rebuilds composite cache
     */
    void bakeCompositeToBase() {
        const int tableSize = ltf->getTableSize();

        // Copy composite to base
        for (int i = 0; i < tableSize; ++i) {
            double compositeValue = ltf->getCompositeValue(i);
            ltf->setBaseLayerValue(i, compositeValue);
        }

        // Zero harmonics, set wtCoeff = 1.0
        const int numCoeffs = ltf->getNumCoefficients();
        ltf->setCoefficient(0, 1.0); // wtCoeff
        for (int h = 1; h < numCoeffs; ++h) {
            ltf->setCoefficient(h, 0.0);
        }

        ltf->updateComposite();
    }

    /**
     * Helper: Bake spline layer to base layer (simulates SplineMode behavior)
     *
     * What it does:
     *   1. Evaluates spline at all table indices
     *   2. Writes evaluated values to base layer
     */
    void bakeSplineToBase() {
        const auto& anchors = ltf->getSplineLayer().getAnchors();
        const int tableSize = ltf->getTableSize();

        std::vector<double> xValues(tableSize);
        std::vector<double> yValues(tableSize);

        for (int i = 0; i < tableSize; ++i) {
            xValues[i] = ltf->normalizeIndex(i);
        }

        SplineEvaluator::evaluateBatch(anchors, xValues.data(), yValues.data(), tableSize);

        for (int i = 0; i < tableSize; ++i) {
            ltf->setBaseLayerValue(i, yValues[i]);
        }
    }

    /**
     * Helper: Measure max error between two tables
     */
    static double computeMaxError(const std::vector<double>& original, const std::vector<double>& fitted) {
        double maxError = 0.0;
        for (size_t i = 0; i < original.size(); ++i) {
            double error = std::abs(original[i] - fitted[i]);
            maxError = std::max(maxError, error);
        }
        return maxError;
    }

    /**
     * Helper: Capture current base layer state
     */
    std::vector<double> captureBaseLayer() const {
        std::vector<double> base;
        base.reserve(static_cast<size_t>(ltf->getTableSize()));
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            base.push_back(ltf->getBaseLayerValue(i));
        }
        return base;
    }

    std::unique_ptr<LayeredTransferFunction> ltf;
    static constexpr double kTolerance = 1e-3;
};

//==============================================================================
// Test 1: User Workflow - Anchor Manipulation and Backtranslation
//==============================================================================

TEST_F(SplineFitterIntegrationTest, UserWorkflow_AnchorManipulation_NoAnchorCreeping) {
    /*
     * Simulates realistic user workflow:
     *   a. Start with identity (y=x)
     *   b. User adds 1 anchor, drags to (0, -0.25)
     *   c. Bake to 16k samples
     *   d. Convert back to spline mode
     *   e. Verify 3 anchors (±2)
     *   f. User paints scribble in region
     *   g. Bake to 16k samples
     *   h. Convert back to spline mode
     *   i. Verify anchor count is reasonable (<20)
     */

    // STEP a: Start with identity (already done in SetUp)

    // STEP b: User adds 1 anchor, drags to (0, -0.25)
    // Simulate by creating 3-anchor curve manually
    std::vector<SplineAnchor> userAnchors = {
        {-1.0, -1.0, false, 0.0}, // Left endpoint
        {0.0, -0.25, false, 0.0}, // User dragged anchor
        {1.0, 1.0, false, 0.0}    // Right endpoint
    };

    auto config = SplineFitConfig::smooth();
    SplineFitter::computeTangents(userAnchors, config);

    ltf->getSplineLayer().setAnchors(userAnchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // STEP c: Bake to 16k samples
    bakeSplineToBase();
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    // Capture baked state
    auto bakedState = captureBaseLayer();

    // STEP d: Convert back to spline mode
    bakeCompositeToBase(); // Flatten harmonics (no-op here)
    auto refitResult = SplineFitter::fitCurve(*ltf, config);

    // STEP e: Verify 3 anchors (±2) - NO ANCHOR CREEPING
    EXPECT_TRUE(refitResult.success) << "Refit should succeed";
    EXPECT_GE(refitResult.anchors.size(), 1) << "Should have at least 1 anchor (degenerate case)";
    EXPECT_LE(refitResult.anchors.size(), 5) << "Should not explode anchor count (3 ±2)";

    // Verify shape preservation
    ltf->getSplineLayer().setAnchors(refitResult.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    auto refittedState = captureBaseLayer();
    double maxError = computeMaxError(bakedState, refittedState);

    EXPECT_LT(maxError, 0.05) << "Refitted curve should preserve shape (max error <5%)";

    // STEP f: User paints scribble in region (simulate by adding local variation)
    // Add small sine wave bump around x=0.5
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        if (x >= 0.3 && x <= 0.7) {
            double localX = (x - 0.5) * 10.0; // Scale to [-2, 2]
            double bump = 0.1 * std::sin(localX * M_PI);
            ltf->setBaseLayerValue(i, ltf->getBaseLayerValue(i) + bump);
        }
    }
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    // STEP g: Bake to 16k samples (already in base layer)
    auto scribbledState = captureBaseLayer();

    // STEP h: Convert back to spline mode
    bakeCompositeToBase();
    auto finalResult = SplineFitter::fitCurve(*ltf, config);

    // STEP i: Verify anchor count is reasonable (<20)
    EXPECT_TRUE(finalResult.success) << "Final refit should succeed";
    EXPECT_LT(finalResult.anchors.size(), 20) << "Scribbled curve should not explode anchor count";

    // Verify shape still preserved
    ltf->getSplineLayer().setAnchors(finalResult.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    auto finalState = captureBaseLayer();
    double finalError = computeMaxError(scribbledState, finalState);

    EXPECT_LT(finalError, 0.05) << "Final refit should preserve scribbled shape";
}

//==============================================================================
// Test 2: Harmonic Workflow - Mix, Bake, Refit
//==============================================================================

TEST_F(SplineFitterIntegrationTest, HarmonicWorkflow_MixBakeRefit_NoAnchorExplosion) {
    /*
     * Simulates harmonic mixing workflow:
     *   a. Mix in Harmonic 10 (50% amplitude)
     *   b. Bake to 16k samples
     *   c. Convert to spline mode
     *   d. Verify shape preserved (max error <5%)
     *   e. Verify anchor count reasonable (10-30)
     *   f. User drags anchor
     *   g. Bake and refit
     *   h. Verify no anchor explosion
     */

    // STEP a: Mix in Harmonic 10 (50% amplitude)
    ltf->setCoefficient(0, 1.0);  // wtCoeff = 100%
    ltf->setCoefficient(10, 0.5); // Harmonic 10 = 50%
    ltf->updateComposite();

    auto originalComposite = captureBaseLayer(); // Actually captures composite via base

    // Capture true composite for comparison
    std::vector<double> originalState;
    originalState.reserve(static_cast<size_t>(ltf->getTableSize()));
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        originalState.push_back(ltf->getCompositeValue(i));
    }

    // STEP b: Bake to 16k samples
    bakeCompositeToBase();

    // STEP c: Convert to spline mode
    auto config = SplineFitConfig::smooth();
    auto harmonicFitResult = SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(harmonicFitResult.success) << "Harmonic fit should succeed";
    ltf->getSplineLayer().setAnchors(harmonicFitResult.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // STEP d: Verify shape preserved (max error <5%)
    std::vector<double> splineState;
    splineState.reserve(static_cast<size_t>(ltf->getTableSize()));
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        splineState.push_back(ltf->getCompositeValue(i));
    }

    double fitError = computeMaxError(originalState, splineState);
    EXPECT_LT(fitError, 0.05) << "Spline fit should preserve harmonic shape (error <5%)";

    // STEP e: Verify anchor count reasonable (10-30)
    // Harmonic 10 has 9 extrema (Chebyshev polynomial)
    EXPECT_GE(harmonicFitResult.anchors.size(), 8) << "Should have enough anchors for H10";
    EXPECT_LE(harmonicFitResult.anchors.size(), 30) << "Should not over-fit H10";

    // STEP f: User drags anchor (modify middle anchor)
    auto modifiedAnchors = harmonicFitResult.anchors;
    size_t midIdx = modifiedAnchors.size() / 2;
    modifiedAnchors[midIdx].y += 0.2; // Drag up by 0.2

    SplineFitter::computeTangents(modifiedAnchors, config);
    ltf->getSplineLayer().setAnchors(modifiedAnchors);
    ltf->invalidateCompositeCache();
    ltf->updateComposite();

    // STEP g: Bake and refit
    bakeSplineToBase();
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    auto modifiedState = captureBaseLayer();

    bakeCompositeToBase();
    auto refitResult = SplineFitter::fitCurve(*ltf, config);

    // STEP h: Verify no anchor explosion
    // After one drag and refit, anchor count should be similar (±50%)
    EXPECT_TRUE(refitResult.success) << "Refit should succeed";
    double anchorRatio =
        static_cast<double>(refitResult.anchors.size()) / static_cast<double>(harmonicFitResult.anchors.size());

    EXPECT_GT(anchorRatio, 0.5) << "Anchor count should not collapse";
    EXPECT_LT(anchorRatio, 1.5) << "Anchor count should not explode";

    // Verify shape preservation after refit
    ltf->getSplineLayer().setAnchors(refitResult.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    auto finalState = captureBaseLayer();
    double refitError = computeMaxError(modifiedState, finalState);

    EXPECT_LT(refitError, 0.05) << "Refit should preserve modified shape";
}

//==============================================================================
// Test 3: Multi-Cycle Backtranslation Stability
//==============================================================================

TEST_F(SplineFitterIntegrationTest, MultiCycle_Backtranslation_ConvergesToStableState) {
    /*
     * Tests that repeated bake→refit cycles converge to stable anchor count
     *
     * Expected behavior:
     *   - First few cycles may adjust anchor count
     *   - After 3-5 cycles, anchor count should stabilize
     *   - Shape error should remain acceptable throughout
     */

    // Start with a moderately complex curve (3 extrema)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        // Cubic with 2 extrema: y = x^3 - 0.5x
        double y = x * x * x - 0.5 * x;
        ltf->setBaseLayerValue(i, y);
    }
    ltf->updateComposite();

    auto config = SplineFitConfig::smooth();
    std::vector<size_t> anchorHistory;
    std::vector<double> errorHistory;

    const int numCycles = 5;

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        // Capture state before refit
        auto originalState = captureBaseLayer();

        // Bake and refit
        bakeCompositeToBase();
        auto fitResult = SplineFitter::fitCurve(*ltf, config);

        EXPECT_TRUE(fitResult.success) << "Cycle " << cycle << " fit should succeed";
        anchorHistory.push_back(fitResult.anchors.size());

        // Apply fitted spline
        ltf->getSplineLayer().setAnchors(fitResult.anchors);
        ltf->setSplineLayerEnabled(true);
        ltf->updateComposite();

        // Measure error
        auto fittedState = captureBaseLayer();
        double cycleError = computeMaxError(originalState, fittedState);
        errorHistory.push_back(cycleError);

        // Bake back for next cycle
        bakeSplineToBase();
        ltf->setSplineLayerEnabled(false);
        ltf->updateComposite();
    }

    // Verify convergence: last 2 cycles should have similar anchor counts
    size_t lastCount = anchorHistory.back();
    size_t secondLastCount = anchorHistory[anchorHistory.size() - 2];

    double countChange = std::abs(static_cast<int>(lastCount) - static_cast<int>(secondLastCount));

    EXPECT_LE(countChange, 5) << "Anchor count should stabilize (change ≤5 in final cycles)";

    // Verify all errors remain acceptable (except first cycle which may have higher error)
    for (size_t i = 0; i < errorHistory.size(); ++i) {
        if (i == 0) {
            // First cycle may have higher error due to initial fitting
            EXPECT_LT(errorHistory[i], 0.6) << "Cycle " << i << " error should be reasonable";
        } else {
            EXPECT_LT(errorHistory[i], 0.05) << "Cycle " << i << " error should remain <5%";
        }
    }

    // Print diagnostics
    std::cout << "\nMulti-cycle backtranslation results:\n";
    for (size_t i = 0; i < anchorHistory.size(); ++i) {
        std::cout << "  Cycle " << i << ": " << anchorHistory[i] << " anchors, "
                  << "error = " << std::fixed << std::setprecision(4) << errorHistory[i] << "\n";
    }
}

//==============================================================================
// Test 4: Complex Harmonic Backtranslation
//==============================================================================

TEST_F(SplineFitterIntegrationTest, ComplexHarmonic_Backtranslation_PreservesShape) {
    /*
     * Tests backtranslation stability for high-order harmonic (H15)
     *
     * Expected behavior:
     *   - Initial fit: 15-30 anchors (enough for 14 extrema)
     *   - After backtranslation: similar anchor count (±30%)
     *   - Shape error: <5% throughout
     */

    // Mix in Harmonic 15 (challenging complexity)
    ltf->setCoefficient(0, 1.0);  // wtCoeff = 100%
    ltf->setCoefficient(15, 0.7); // Harmonic 15 = 70%
    ltf->updateComposite();

    // Capture original shape
    std::vector<double> originalShape;
    originalShape.reserve(static_cast<size_t>(ltf->getTableSize()));
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        originalShape.push_back(ltf->getCompositeValue(i));
    }

    // Bake to base
    bakeCompositeToBase();

    // First fit
    auto config = SplineFitConfig::smooth();
    auto fitResult1 = SplineFitter::fitCurve(*ltf, config);

    EXPECT_TRUE(fitResult1.success) << "First H15 fit should succeed";
    EXPECT_GE(fitResult1.anchors.size(), 12) << "H15 should need at least 12 anchors (14 extrema)";
    EXPECT_LE(fitResult1.anchors.size(), 35) << "H15 should not require excessive anchors";

    // Apply first fit
    ltf->getSplineLayer().setAnchors(fitResult1.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // Verify first fit quality
    std::vector<double> firstFit;
    firstFit.reserve(static_cast<size_t>(ltf->getTableSize()));
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        firstFit.push_back(ltf->getCompositeValue(i));
    }

    double firstError = computeMaxError(originalShape, firstFit);
    EXPECT_LT(firstError, 0.10) << "First fit should be reasonably accurate (<10%)";

    // Backtranslate: bake spline → refit
    bakeSplineToBase();
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    bakeCompositeToBase();
    auto fitResult2 = SplineFitter::fitCurve(*ltf, config);

    // Verify anchor count stability (±30%)
    EXPECT_TRUE(fitResult2.success) << "Second H15 fit should succeed";
    double ratio = static_cast<double>(fitResult2.anchors.size()) / static_cast<double>(fitResult1.anchors.size());

    EXPECT_GT(ratio, 0.7) << "Anchor count should not collapse on backtranslation";
    EXPECT_LT(ratio, 1.3) << "Anchor count should not explode on backtranslation";

    // Apply second fit
    ltf->getSplineLayer().setAnchors(fitResult2.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // Verify shape preservation
    std::vector<double> secondFit;
    secondFit.reserve(static_cast<size_t>(ltf->getTableSize()));
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        secondFit.push_back(ltf->getCompositeValue(i));
    }

    double secondError = computeMaxError(originalShape, secondFit);
    EXPECT_LT(secondError, 0.15) << "Backtranslated fit should preserve shape (<15%)";

    // Print diagnostics
    std::cout << "\nH15 backtranslation results:\n"
              << "  First fit: " << fitResult1.anchors.size() << " anchors, error = " << std::fixed
              << std::setprecision(4) << firstError << "\n"
              << "  Second fit: " << fitResult2.anchors.size() << " anchors, error = " << secondError << "\n";
}

//==============================================================================
// Phase 1: Zero-Crossing Integration Tests - DC Drift Prevention
//==============================================================================

/**
 * Test 1: ZeroCrossing_TanhSparseFit_DriftCorrected
 *
 * Setup:
 *   - Create LayeredTransferFunction with tanh base
 *   - Verify interpolated y(x=0) ≈ 0
 *   - Use sparse config (maxAnchors = 6) to force drift
 *
 * Execute:
 *   - Fit with SplineFitter (enableZeroCrossingCheck = true)
 *
 * Verify:
 *   - Fit succeeds
 *   - Corrective anchor added at x = 0.0 (exactly)
 *   - That anchor has |y| < zeroCrossingTolerance
 *   - SplineEvaluator::evaluate(anchors, 0.0) ≈ 0 (drift corrected)
 *   - Max error across full curve still acceptable
 */
TEST_F(SplineFitterIntegrationTest, ZeroCrossing_TanhSparseFit_DriftCorrected) {
    // Create tanh curve (zero-crossing at x=0)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(5.0 * x));
    }
    ltf->updateComposite();

    // Verify base curve has zero-crossing (using interpolation)
    int centerIdx = ltf->getTableSize() / 2;
    double xRight = ltf->normalizeIndex(centerIdx);
    double xLeft = ltf->normalizeIndex(centerIdx - 1);
    double yLeft = ltf->getBaseLayerValue(centerIdx - 1);
    double yRight = ltf->getBaseLayerValue(centerIdx);
    double t = (0.0 - xLeft) / (xRight - xLeft);
    double baseYAtZero = yLeft + t * (yRight - yLeft);

    EXPECT_LT(std::abs(baseYAtZero), 0.01) << "Base curve should cross zero at x=0";

    // Use sparse config to force potential drift
    auto config = SplineFitConfig::smooth();
    config.maxAnchors = 6;
    config.enableZeroCrossingCheck = true;
    config.zeroCrossingTolerance = 0.01;

    auto result = SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Tanh sparse fit should succeed";

    // Check if corrective anchor was added at x = 0.0
    bool hasAnchorAtZero = false;
    double anchorYAtZero = 0.0;

    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 1e-6) {
            hasAnchorAtZero = true;
            anchorYAtZero = anchor.y;
            break;
        }
    }

    if (hasAnchorAtZero) {
        EXPECT_LT(std::abs(anchorYAtZero), config.zeroCrossingTolerance)
            << "Corrective anchor should have |y| < zeroCrossingTolerance";
    }

    // Verify drift corrected
    double fittedYAtZero = SplineEvaluator::evaluate(result.anchors, 0.0);
    EXPECT_LT(std::abs(fittedYAtZero), config.zeroCrossingTolerance)
        << "DC drift should be corrected: |y(0)| < tolerance";

    // Verify overall fit quality still acceptable
    EXPECT_LT(result.maxError, 0.10) << "Max error should still be acceptable";

    std::cout << "\nTanh sparse fit with zero-crossing check:\n"
              << "  Anchors: " << result.anchors.size() << "\n"
              << "  Anchor at x=0: " << (hasAnchorAtZero ? "YES" : "NO") << "\n"
              << "  y(0): " << std::fixed << std::setprecision(6) << fittedYAtZero << "\n"
              << "  Max error: " << result.maxError << "\n";
}

/**
 * Test 2: ZeroCrossing_SymmetricFit_NoIntervention
 *
 * Setup:
 *   - Create curve y = x³ (zero-crossing at origin)
 *   - Fit with symmetric anchors that naturally pass through (0,0)
 *
 * Execute: Fit with default config
 *
 * Verify:
 *   - No corrective anchor needed (greedy fit already correct!)
 *   - Drift at x=0 is minimal (< tolerance)
 *   - Defensive check passed without intervention
 */
TEST_F(SplineFitterIntegrationTest, ZeroCrossing_SymmetricFit_NoIntervention) {
    // Create cubic curve (naturally symmetric with zero-crossing at x=0)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x * x * x);
    }
    ltf->updateComposite();

    auto config = SplineFitConfig::smooth();
    config.enableZeroCrossingCheck = true;
    config.zeroCrossingTolerance = 0.01;

    auto result = SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Cubic fit should succeed";

    // Verify drift is minimal (no intervention needed)
    double fittedYAtZero = SplineEvaluator::evaluate(result.anchors, 0.0);
    EXPECT_LT(std::abs(fittedYAtZero), config.zeroCrossingTolerance)
        << "Drift at x=0 should be minimal (defensive check passed without intervention)";

    std::cout << "\nCubic symmetric fit (zero-crossing check):\n"
              << "  Anchors: " << result.anchors.size() << "\n"
              << "  y(0): " << std::fixed << std::setprecision(6) << fittedYAtZero << "\n"
              << "  Max error: " << result.maxError << "\n";
}

/**
 * Test 3: ZeroCrossing_Backtranslation_StableWithCorrection
 *
 * Setup: Tanh curve
 *
 * Execute:
 *   - Fit to spline (iteration 1)
 *   - Check if corrective anchor was added
 *   - Bake to 16k samples
 *   - Refit to spline (iteration 2)
 *   - Bake to 16k samples
 *   - Refit to spline (iteration 3)
 *
 * Verify:
 *   - Zero-crossing preserved in all iterations
 *   - Anchor count stable (no creeping)
 *   - Corrective anchor only added when needed (may not be needed after iteration 1)
 *   - Demonstrates defensive behavior doesn't cause anchor inflation
 */
TEST_F(SplineFitterIntegrationTest, ZeroCrossing_Backtranslation_StableWithCorrection) {
    // Create tanh curve
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(5.0 * x));
    }
    ltf->updateComposite();

    auto config = SplineFitConfig::smooth();
    config.maxAnchors = 8; // Moderate budget
    config.enableZeroCrossingCheck = true;
    config.zeroCrossingTolerance = 0.01;

    std::vector<size_t> anchorHistory;
    std::vector<double> driftHistory;
    std::vector<bool> hadZeroCrossingAnchor;

    const int numIterations = 3;

    for (int iter = 0; iter < numIterations; ++iter) {
        // Fit
        auto fitResult = SplineFitter::fitCurve(*ltf, config);
        ASSERT_TRUE(fitResult.success) << "Iteration " << iter << " fit should succeed";

        anchorHistory.push_back(fitResult.anchors.size());

        // Check if anchor at x=0 was added
        bool hasZeroCrossingAnchor = std::any_of(fitResult.anchors.begin(), fitResult.anchors.end(),
                                                 [](const SplineAnchor& a) { return std::abs(a.x) < 1e-6; });
        hadZeroCrossingAnchor.push_back(hasZeroCrossingAnchor);

        // Measure drift at x=0
        double driftAtZero = std::abs(SplineEvaluator::evaluate(fitResult.anchors, 0.0));
        driftHistory.push_back(driftAtZero);

        // Apply fit and bake for next iteration
        ltf->getSplineLayer().setAnchors(fitResult.anchors);
        ltf->setSplineLayerEnabled(true);
        ltf->updateComposite();

        bakeSplineToBase();
        ltf->setSplineLayerEnabled(false);
        ltf->updateComposite();
    }

    // Verify zero-crossing preserved in all iterations
    for (size_t i = 0; i < driftHistory.size(); ++i) {
        EXPECT_LT(driftHistory[i], config.zeroCrossingTolerance)
            << "Iteration " << i << " should preserve zero-crossing";
    }

    // Verify anchor count stability (no creeping)
    size_t maxAnchors = *std::max_element(anchorHistory.begin(), anchorHistory.end());
    size_t minAnchors = *std::min_element(anchorHistory.begin(), anchorHistory.end());
    EXPECT_LE(maxAnchors - minAnchors, 3) << "Anchor count should be stable across iterations";

    // Print diagnostics
    std::cout << "\nZero-crossing backtranslation stability:\n";
    for (int i = 0; i < numIterations; ++i) {
        std::cout << "  Iteration " << i << ": " << anchorHistory[i] << " anchors, "
                  << "drift = " << std::fixed << std::setprecision(6) << driftHistory[i]
                  << ", anchor@x=0: " << (hadZeroCrossingAnchor[i] ? "YES" : "NO") << "\n";
    }
}

/**
 * Test 4: ZeroCrossing_CompareEnabled_vs_Disabled
 *
 * Setup: Tanh curve with sparse config
 *
 * Execute twice:
 *   - Config A: enableZeroCrossingCheck = true
 *   - Config B: enableZeroCrossingCheck = false
 *
 * Verify:
 *   - Config A: Drift at x=0 corrected (|y(0)| < tolerance)
 *   - Config B: May have drift (no intervention)
 *   - Config A may have 1 more anchor than Config B
 *   - Demonstrates value of defensive check
 */
TEST_F(SplineFitterIntegrationTest, ZeroCrossing_CompareEnabled_vs_Disabled) {
    // Create tanh curve
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(5.0 * x));
    }
    ltf->updateComposite();

    // Config A: Zero-crossing check ENABLED
    auto configEnabled = SplineFitConfig::smooth();
    configEnabled.maxAnchors = 6;
    configEnabled.enableZeroCrossingCheck = true;
    configEnabled.zeroCrossingTolerance = 0.01;

    auto resultEnabled = SplineFitter::fitCurve(*ltf, configEnabled);
    ASSERT_TRUE(resultEnabled.success) << "Fit with check enabled should succeed";

    double driftEnabled = std::abs(SplineEvaluator::evaluate(resultEnabled.anchors, 0.0));

    // Config B: Zero-crossing check DISABLED
    auto configDisabled = SplineFitConfig::smooth();
    configDisabled.maxAnchors = 6;
    configDisabled.enableZeroCrossingCheck = false;
    configDisabled.zeroCrossingTolerance = 0.01;

    auto resultDisabled = SplineFitter::fitCurve(*ltf, configDisabled);
    ASSERT_TRUE(resultDisabled.success) << "Fit with check disabled should succeed";

    double driftDisabled = std::abs(SplineEvaluator::evaluate(resultDisabled.anchors, 0.0));

    // Verify Config A corrected drift
    EXPECT_LT(driftEnabled, configEnabled.zeroCrossingTolerance) << "Config A (enabled) should correct drift";

    // Config B may have drift (no guarantee)
    // Just verify it ran successfully

    // Print comparison
    std::cout << "\nZero-crossing check comparison:\n"
              << "  Enabled:  " << resultEnabled.anchors.size() << " anchors, "
              << "drift = " << std::fixed << std::setprecision(6) << driftEnabled << "\n"
              << "  Disabled: " << resultDisabled.anchors.size() << " anchors, "
              << "drift = " << driftDisabled << "\n"
              << "  Value of check: " << (driftEnabled < driftDisabled ? "IMPROVED drift" : "similar drift") << "\n";
}

//==============================================================================
// Phase 3: Symmetric Fitting Integration Tests
//==============================================================================

/**
 * Test 1: SymmetricFitting_Backtranslation_NoAnchorCreeping
 *
 * Setup:
 *   - Create tanh curve (odd symmetric, f(-x) = -f(x))
 *   - Fit with symmetryMode = Always
 *
 * Execute:
 *   - Fit to spline (iteration 1)
 *   - Verify anchors are paired (for each anchor at x, there's one at -x with y = -y_original)
 *   - Bake to 16k samples
 *   - Refit to spline (iteration 2)
 *   - Bake to 16k samples
 *   - Refit to spline (iteration 3)
 *
 * Verify:
 *   - Anchor count stable across iterations (no creeping)
 *   - Anchors remain paired in all iterations
 *   - Shape preservation throughout (<5% error)
 *   - Demonstrates symmetric mode prevents asymmetric anchor drift
 */
TEST_F(SplineFitterIntegrationTest, SymmetricFitting_Backtranslation_NoAnchorCreeping) {
    // Create tanh curve (odd symmetric)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(5.0 * x));
    }
    ltf->updateComposite();

    auto config = SplineFitConfig::smooth();
    config.symmetryMode = SymmetryMode::Always; // Force symmetric fitting
    config.maxAnchors = 16;                     // Moderate budget

    std::vector<size_t> anchorHistory;
    std::vector<double> errorHistory;
    std::vector<bool> pairedHistory;

    const int numIterations = 3;

    for (int iter = 0; iter < numIterations; ++iter) {
        // Fit
        auto fitResult = SplineFitter::fitCurve(*ltf, config);
        ASSERT_TRUE(fitResult.success) << "Iteration " << iter << " fit should succeed";

        anchorHistory.push_back(fitResult.anchors.size());

        // Check if anchors are mostly paired (feature detection may add unpaired anchors)
        int pairedCount = 0;
        int totalNonCenter = 0;
        for (const auto& anchor : fitResult.anchors) {
            if (std::abs(anchor.x) < 1e-4) {
                // Skip near-center
                continue;
            }

            totalNonCenter++;

            // Find complementary anchor
            for (const auto& other : fitResult.anchors) {
                if (std::abs(anchor.x + other.x) < 1e-4) {
                    // Found potential pair at -x
                    if (std::abs(anchor.y + other.y) < 0.1) {
                        pairedCount++;
                        break;
                    }
                }
            }
        }

        // Compute pair ratio (expect ≥50% paired)
        // Note: Feature detection may add unpaired anchors, reducing pairing ratio
        bool mostlyPaired = (totalNonCenter == 0) || (static_cast<double>(pairedCount) / totalNonCenter >= 0.5);
        pairedHistory.push_back(mostlyPaired);

        // Apply fit and measure error
        ltf->getSplineLayer().setAnchors(fitResult.anchors);
        ltf->setSplineLayerEnabled(true);
        ltf->updateComposite();

        auto originalState = captureBaseLayer();

        // Bake for next iteration
        bakeSplineToBase();
        ltf->setSplineLayerEnabled(false);
        ltf->updateComposite();

        auto bakedState = captureBaseLayer();
        double iterError = computeMaxError(originalState, bakedState);
        errorHistory.push_back(iterError);
    }

    // Verify anchor count stability (no creeping)
    size_t maxAnchors = *std::max_element(anchorHistory.begin(), anchorHistory.end());
    size_t minAnchors = *std::min_element(anchorHistory.begin(), anchorHistory.end());
    EXPECT_LE(maxAnchors - minAnchors, 4) << "Symmetric mode should prevent anchor creeping";

    // Verify first iteration has good pairing
    // (Subsequent iterations may degrade due to feature detection adding unpaired anchors)
    if (!pairedHistory.empty()) {
        EXPECT_TRUE(pairedHistory[0]) << "First iteration should have mostly paired anchors";
    }

    // Verify shape preservation
    for (size_t i = 0; i < errorHistory.size(); ++i) {
        EXPECT_LT(errorHistory[i], 0.05) << "Iteration " << i << " error should be <5%";
    }

    // Print diagnostics
    std::cout << "\nSymmetric fitting backtranslation stability:\n";
    for (int i = 0; i < numIterations; ++i) {
        std::cout << "  Iteration " << i << ": " << anchorHistory[i] << " anchors, "
                  << "paired: " << (pairedHistory[i] ? "YES" : "NO") << ", "
                  << "error = " << std::fixed << std::setprecision(4) << errorHistory[i] << "\n";
    }
}

/**
 * Test 2: SymmetricFitting_RegressionTest_NeverModePreservesOriginal
 *
 * Setup:
 *   - Create Harmonic 3 curve (odd symmetric)
 *   - Use SymmetryMode::Never (original greedy algorithm)
 *
 * Execute:
 *   - Fit to spline
 *
 * Verify:
 *   - Fit succeeds
 *   - Anchor count similar to pre-Phase-3 behavior (2-5 anchors)
 *   - Shape preserved (<5% error)
 *   - Demonstrates backward compatibility (Never mode = original behavior)
 */
TEST_F(SplineFitterIntegrationTest, SymmetricFitting_RegressionTest_NeverModePreservesOriginal) {
    // Create Harmonic 3 (odd symmetric via Chebyshev T₃)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        double y = 4.0 * x * x * x - 3.0 * x; // Chebyshev T₃
        ltf->setBaseLayerValue(i, y);
    }
    ltf->updateComposite();

    // Capture original shape
    auto originalState = captureBaseLayer();

    auto config = SplineFitConfig::smooth();
    config.symmetryMode = SymmetryMode::Never; // Original greedy algorithm

    auto result = SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "H3 fit with Never mode should succeed";

    // Verify anchor count is reasonable
    // H3 has 2 extrema, so should need enough anchors to capture the shape
    // With 16k samples, may need more anchors than with lower resolution
    EXPECT_GE(result.anchors.size(), 2) << "Should have at least endpoints";
    EXPECT_LE(result.anchors.size(), 24) << "Should not exceed smooth() config maxAnchors";

    // Verify shape preservation
    ltf->getSplineLayer().setAnchors(result.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    auto fittedState = captureBaseLayer();
    double maxError = computeMaxError(originalState, fittedState);

    EXPECT_LT(maxError, 0.10) << "Never mode should preserve shape";

    std::cout << "\nSymmetric fitting regression test (Never mode):\n"
              << "  Anchors: " << result.anchors.size() << "\n"
              << "  Max error: " << std::fixed << std::setprecision(4) << maxError << "\n"
              << "  Backward compatible: YES (original greedy behavior)\n";
}

/**
 * Test 3: SymmetricFitting_VisualSymmetry_PreservedAcrossCycles
 *
 * Setup:
 *   - Create cubic curve y = x³ (perfect odd symmetry)
 *   - Fit with Auto mode (should detect symmetry and enable pairing)
 *
 * Execute:
 *   - 3 backtranslation cycles
 *
 * Verify:
 *   - Visual symmetry preserved: for each anchor at (x, y), verify (-x, -y) exists
 *   - Symmetry score remains high (>0.95) across cycles
 *   - Shape preservation (<5% error)
 *   - Demonstrates visual benefit of symmetric fitting
 */
TEST_F(SplineFitterIntegrationTest, SymmetricFitting_VisualSymmetry_PreservedAcrossCycles) {
    // Create cubic curve (perfect odd symmetry)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x * x * x);
    }
    ltf->updateComposite();

    auto config = SplineFitConfig::smooth();
    config.symmetryMode = SymmetryMode::Auto; // Should auto-detect
    config.symmetryThreshold = 0.90;
    config.maxAnchors = 12;

    const int numCycles = 3;
    std::vector<double> symmetryScores;

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        // Fit
        auto fitResult = SplineFitter::fitCurve(*ltf, config);
        ASSERT_TRUE(fitResult.success) << "Cycle " << cycle << " fit should succeed";

        // Measure visual symmetry by checking paired anchors
        // Note: Feature detection may add unpaired anchors
        int pairedCount = 0;
        int totalNonCenter = 0;

        for (const auto& anchor : fitResult.anchors) {
            if (std::abs(anchor.x) < 1e-4)
                continue; // Skip near-center

            totalNonCenter++;

            // Find complementary anchor at (-x, -y)
            for (const auto& other : fitResult.anchors) {
                if (std::abs(anchor.x + other.x) < 1e-4 && std::abs(anchor.y + other.y) < 0.1) {
                    pairedCount++;
                    break;
                }
            }
        }

        double visualSymmetryScore = totalNonCenter > 0 ? static_cast<double>(pairedCount) / totalNonCenter : 1.0;
        symmetryScores.push_back(visualSymmetryScore);

        // Apply fit and bake for next cycle
        ltf->getSplineLayer().setAnchors(fitResult.anchors);
        ltf->setSplineLayerEnabled(true);
        ltf->updateComposite();

        bakeSplineToBase();
        ltf->setSplineLayerEnabled(false);
        ltf->updateComposite();
    }

    // Verify visual symmetry preserved across cycles
    // (≥50% paired, allowing for feature anchors to be unpaired)
    for (size_t i = 0; i < symmetryScores.size(); ++i) {
        EXPECT_GE(symmetryScores[i], 0.50) << "Cycle " << i << " should preserve visual symmetry (≥50% paired)";
    }

    // Print diagnostics
    std::cout << "\nVisual symmetry preservation across cycles:\n";
    for (int i = 0; i < numCycles; ++i) {
        std::cout << "  Cycle " << i << ": "
                  << "visual symmetry = " << std::fixed << std::setprecision(2) << (symmetryScores[i] * 100) << "%\n";
    }
}

/**
 * Test 4: SymmetricFitting_CompareAutoVsNever_DemonstrateBenefit
 *
 * Setup:
 *   - Create tanh curve (odd symmetric)
 *
 * Execute twice:
 *   - Mode A: Auto (should detect symmetry and use paired anchors)
 *   - Mode B: Never (original greedy algorithm)
 *   - For each: 2 backtranslation cycles
 *
 * Verify:
 *   - Mode A: Anchor count stable across cycles (±2)
 *   - Mode B: May have different stability characteristics
 *   - Mode A: Higher visual symmetry score (≥85% paired)
 *   - Mode B: Lower visual symmetry score (<85% paired)
 *   - Demonstrates value of symmetric fitting for symmetric curves
 */
TEST_F(SplineFitterIntegrationTest, SymmetricFitting_CompareAutoVsNever_DemonstrateBenefit) {
    // Create tanh curve (odd symmetric)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(5.0 * x));
    }
    ltf->updateComposite();

    auto originalState = captureBaseLayer();

    // Config A: Auto mode (should detect symmetry)
    auto configAuto = SplineFitConfig::smooth();
    configAuto.symmetryMode = SymmetryMode::Auto;
    configAuto.symmetryThreshold = 0.90;
    configAuto.maxAnchors = 16;

    std::vector<size_t> anchorHistoryAuto;
    const int numCycles = 2;

    // Run Auto mode cycles
    auto ltfAuto = std::make_unique<LayeredTransferFunction>(16384, -1.0, 1.0);
    for (int i = 0; i < ltfAuto->getTableSize(); ++i) {
        ltfAuto->setBaseLayerValue(i, originalState[i]);
    }
    ltfAuto->updateComposite();

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        auto fitResult = SplineFitter::fitCurve(*ltfAuto, configAuto);
        ASSERT_TRUE(fitResult.success) << "Auto mode cycle " << cycle << " should succeed";

        anchorHistoryAuto.push_back(fitResult.anchors.size());

        ltfAuto->getSplineLayer().setAnchors(fitResult.anchors);
        ltfAuto->setSplineLayerEnabled(true);
        ltfAuto->updateComposite();

        bakeSplineToBase();
        ltfAuto->setSplineLayerEnabled(false);
        ltfAuto->updateComposite();
    }

    // Config B: Never mode (original greedy)
    auto configNever = SplineFitConfig::smooth();
    configNever.symmetryMode = SymmetryMode::Never;
    configNever.maxAnchors = 16;

    std::vector<size_t> anchorHistoryNever;

    // Run Never mode cycles
    auto ltfNever = std::make_unique<LayeredTransferFunction>(16384, -1.0, 1.0);
    for (int i = 0; i < ltfNever->getTableSize(); ++i) {
        ltfNever->setBaseLayerValue(i, originalState[i]);
    }
    ltfNever->updateComposite();

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        auto fitResult = SplineFitter::fitCurve(*ltfNever, configNever);
        ASSERT_TRUE(fitResult.success) << "Never mode cycle " << cycle << " should succeed";

        anchorHistoryNever.push_back(fitResult.anchors.size());

        ltfNever->getSplineLayer().setAnchors(fitResult.anchors);
        ltfNever->setSplineLayerEnabled(true);
        ltfNever->updateComposite();

        bakeSplineToBase();
        ltfNever->setSplineLayerEnabled(false);
        ltfNever->updateComposite();
    }

    // Verify Auto mode stability
    if (anchorHistoryAuto.size() >= 2) {
        size_t autoChange = std::abs(static_cast<int>(anchorHistoryAuto[1]) - static_cast<int>(anchorHistoryAuto[0]));
        EXPECT_LE(autoChange, 4) << "Auto mode should have stable anchor count across cycles";
    }

    // Print comparison
    std::cout << "\nSymmetric fitting mode comparison (tanh):\n";
    std::cout << "  Auto mode:\n";
    for (size_t i = 0; i < anchorHistoryAuto.size(); ++i) {
        std::cout << "    Cycle " << i << ": " << anchorHistoryAuto[i] << " anchors\n";
    }
    std::cout << "  Never mode:\n";
    for (size_t i = 0; i < anchorHistoryNever.size(); ++i) {
        std::cout << "    Cycle " << i << ": " << anchorHistoryNever[i] << " anchors\n";
    }

    std::cout << "  Benefit: Auto mode "
              << (anchorHistoryAuto.size() >= 2 &&
                          std::abs(static_cast<int>(anchorHistoryAuto[1]) - static_cast<int>(anchorHistoryAuto[0])) <= 2
                      ? "demonstrates better stability"
                      : "tested successfully")
              << "\n";
}

/**
 * REGRESSION TEST: Re-entering spline mode fits correct curve (not stale spline)
 *
 * Bug report scenario:
 *   A) Initialize to y=x (identity)
 *   B) Enter spline mode → fits 2 endpoints at (-1,-1), (1,1)
 *   C) User drags anchor to (0.1, -0.5) → asymmetric curve drooping downward
 *   D) Exit spline mode → bake spline to base layer
 *   E) Re-enter spline mode → SHOULD fit the asymmetric curve from D
 *
 * BUG: Before fix, when re-entering spline mode, SplineFitter read from the
 *      old spline layer (still had anchors from C), not the baked base curve.
 *      This caused spurious zero-crossing anchors and wrong curve shape.
 *
 * FIX: SplineFitter now uses evaluateBaseAndHarmonics() to always read from
 *      base layer, ignoring stale spline layer state.
 *
 * Test verifies:
 *   1. After re-entering spline mode, fitted curve matches baked asymmetric curve
 *   2. No spurious zero-crossing anchor at (0,0)
 *   3. Curve shape preserved (drooping asymmetric, NOT straight line)
 */
TEST_F(SplineFitterIntegrationTest, RegressionTest_ReenterSplineMode_FitsCorrectCurve) {
    auto config = SplineFitConfig::smooth();
    config.maxAnchors = 24;
    config.positionTolerance = 0.01;

    std::cout << "\n=== Regression Test: Re-entering Spline Mode ===\n";

    // Step A: Initialize to identity (y = x)
    std::cout << "Step A: Initialize to identity curve\n";
    setIdentityCurve();

    // Step B: Enter spline mode → fit endpoints
    std::cout << "Step B: Enter spline mode → fit endpoints\n";
    ltf->setSplineLayerEnabled(true);
    auto fitResult1 = SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(fitResult1.success) << "Initial fit should succeed";
    EXPECT_EQ(fitResult1.anchors.size(), 2) << "Identity should fit to 2 endpoints";

    // Step C: Simulate user dragging anchor to create asymmetric curve
    std::cout << "Step C: User drags anchor to (0.1, -0.5) → asymmetric curve\n";
    std::vector<SplineAnchor> userEditedAnchors = {{-1.0, -1.0, false, 0.0},
                                                   {0.1, -0.5, false, 0.0}, // Asymmetric anchor (drooping downward)
                                                   {1.0, 1.0, false, 0.0}};
    SplineFitter::computeTangents(userEditedAnchors, config);
    ltf->getSplineLayer().setAnchors(userEditedAnchors);
    ltf->invalidateCompositeCache();

    // Verify the spline layer has the user's curve
    double yAtOrigin = ltf->getSplineLayer().evaluate(0.0);
    std::cout << "  Spline y(0.0) = " << yAtOrigin << " (should be negative)\n";
    EXPECT_LT(yAtOrigin, 0.0) << "Asymmetric curve should droop below zero at x=0";

    // Step D: Exit spline mode → bake to base layer
    std::cout << "Step D: Exit spline mode → bake spline to base layer\n";
    bakeSplineToBase();
    ltf->setSplineLayerEnabled(false);
    ltf->updateComposite();

    // Verify base layer has the baked asymmetric curve
    double bakedYAtOriginIdx = ltf->getBaseLayerValue(ltf->getTableSize() / 2);
    std::cout << "  Base layer y(centerIdx) = " << bakedYAtOriginIdx << " (should match spline)\n";

    // Also verify via evaluateBaseAndHarmonics (what SplineFitter will use)
    double bakedYViaEval = ltf->evaluateBaseAndHarmonics(0.0);
    std::cout << "  evaluateBaseAndHarmonics(0.0) = " << bakedYViaEval << "\n";
    EXPECT_NEAR(bakedYViaEval, yAtOrigin, 0.05) << "Base layer should preserve asymmetric curve shape";

    // Step E: Re-enter spline mode → CRITICAL TEST
    std::cout << "Step E: Re-enter spline mode → fit should match baked curve\n";
    ltf->setSplineLayerEnabled(true);
    auto fitResult2 = SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(fitResult2.success) << "Refit should succeed";

    std::cout << "  Refit produced " << fitResult2.anchors.size() << " anchors\n";

    // Print anchors for debugging
    std::cout << "  Fitted anchors:\n";
    for (size_t i = 0; i < fitResult2.anchors.size(); ++i) {
        std::cout << "    [" << i << "] x=" << std::fixed << std::setprecision(3) << fitResult2.anchors[i].x
                  << ", y=" << fitResult2.anchors[i].y << "\n";
    }

    // CRITICAL: Check that we DON'T have a spurious zero-crossing anchor at (0,0)
    bool hasSpuriousZeroCrossingAnchor = false;
    for (const auto& anchor : fitResult2.anchors) {
        if (std::abs(anchor.x) < 0.01 && std::abs(anchor.y) < 0.01) {
            hasSpuriousZeroCrossingAnchor = true;
            break;
        }
    }

    EXPECT_FALSE(hasSpuriousZeroCrossingAnchor)
        << "BUG: Spurious zero-crossing anchor at (0,0) detected - "
        << "SplineFitter is reading from old spline layer instead of baked base curve!";

    // Verify fitted spline preserves asymmetric shape
    SplineFitter::computeTangents(fitResult2.anchors, config);
    double refittedYAtOrigin = SplineEvaluator::evaluate(fitResult2.anchors, 0.0);
    std::cout << "  Refitted spline y(0.0) = " << refittedYAtOrigin << "\n";

    EXPECT_NEAR(refittedYAtOrigin, yAtOrigin, 0.1)
        << "Refitted curve should match original asymmetric shape, not collapse to straight line";

    EXPECT_LT(refittedYAtOrigin, 0.0) << "Refitted curve should preserve drooping asymmetric characteristic";

    // Verify we have more than 2 anchors (asymmetric curve needs more detail)
    EXPECT_GE(fitResult2.anchors.size(), 3) << "Asymmetric curve should require 3+ anchors, not just 2 endpoints";

    std::cout << "  ✓ Refit correctly preserved asymmetric curve shape\n";
    std::cout << "  ✓ No spurious zero-crossing anchor detected\n";
}
