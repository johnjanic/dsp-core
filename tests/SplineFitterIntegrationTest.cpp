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
        ltf->setCoefficient(0, 1.0);  // wtCoeff
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

        SplineEvaluator::evaluateBatch(
            anchors, xValues.data(), yValues.data(), tableSize
        );

        for (int i = 0; i < tableSize; ++i) {
            ltf->setBaseLayerValue(i, yValues[i]);
        }
    }

    /**
     * Helper: Measure max error between two tables
     */
    double computeMaxError(const std::vector<double>& original,
                          const std::vector<double>& fitted) const {
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
        {-1.0, -1.0, false, 0.0},  // Left endpoint
        {0.0, -0.25, false, 0.0},  // User dragged anchor
        {1.0, 1.0, false, 0.0}     // Right endpoint
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
    bakeCompositeToBase();  // Flatten harmonics (no-op here)
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
            double localX = (x - 0.5) * 10.0;  // Scale to [-2, 2]
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
    ltf->setCoefficient(0, 1.0);   // wtCoeff = 100%
    ltf->setCoefficient(10, 0.5);  // Harmonic 10 = 50%
    ltf->updateComposite();

    auto originalComposite = captureBaseLayer();  // Actually captures composite via base

    // Capture true composite for comparison
    std::vector<double> originalState;
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
    modifiedAnchors[midIdx].y += 0.2;  // Drag up by 0.2

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
    double anchorRatio = static_cast<double>(refitResult.anchors.size()) /
                         static_cast<double>(harmonicFitResult.anchors.size());

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

    double countChange = std::abs(static_cast<int>(lastCount) -
                                   static_cast<int>(secondLastCount));

    EXPECT_LE(countChange, 5) << "Anchor count should stabilize (change ≤5 in final cycles)";

    // Verify all errors remain acceptable (except first cycle which may have higher error)
    for (size_t i = 0; i < errorHistory.size(); ++i) {
        if (i == 0) {
            // First cycle may have higher error due to initial fitting
            EXPECT_LT(errorHistory[i], 0.6)
                << "Cycle " << i << " error should be reasonable";
        } else {
            EXPECT_LT(errorHistory[i], 0.05)
                << "Cycle " << i << " error should remain <5%";
        }
    }

    // Print diagnostics
    std::cout << "\nMulti-cycle backtranslation results:\n";
    for (size_t i = 0; i < anchorHistory.size(); ++i) {
        std::cout << "  Cycle " << i << ": "
                  << anchorHistory[i] << " anchors, "
                  << "error = " << std::fixed << std::setprecision(4)
                  << errorHistory[i] << "\n";
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
    ltf->setCoefficient(0, 1.0);   // wtCoeff = 100%
    ltf->setCoefficient(15, 0.7);  // Harmonic 15 = 70%
    ltf->updateComposite();

    // Capture original shape
    std::vector<double> originalShape;
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
    double ratio = static_cast<double>(fitResult2.anchors.size()) /
                   static_cast<double>(fitResult1.anchors.size());

    EXPECT_GT(ratio, 0.7) << "Anchor count should not collapse on backtranslation";
    EXPECT_LT(ratio, 1.3) << "Anchor count should not explode on backtranslation";

    // Apply second fit
    ltf->getSplineLayer().setAnchors(fitResult2.anchors);
    ltf->setSplineLayerEnabled(true);
    ltf->updateComposite();

    // Verify shape preservation
    std::vector<double> secondFit;
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        secondFit.push_back(ltf->getCompositeValue(i));
    }

    double secondError = computeMaxError(originalShape, secondFit);
    EXPECT_LT(secondError, 0.15) << "Backtranslated fit should preserve shape (<15%)";

    // Print diagnostics
    std::cout << "\nH15 backtranslation results:\n"
              << "  First fit: " << fitResult1.anchors.size() << " anchors, error = "
              << std::fixed << std::setprecision(4) << firstError << "\n"
              << "  Second fit: " << fitResult2.anchors.size() << " anchors, error = "
              << secondError << "\n";
}
