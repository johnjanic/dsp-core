#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <cmath>
#include <iostream>

using namespace dsp_core;

/**
 * Test suite for reproducing Harmonic 40 issue with production table size (16384)
 *
 * Issue: Simple curves (y=x with one anchor added) fit perfectly with adaptive tolerance.
 *        Complex curves (Harmonic 40) get horrible fit with terrible accuracy.
 *
 * Hypothesis: Feature detection or density constraints are blocking anchor placement,
 *             OR complexity score is incorrectly computed for 16k samples.
 */
class Harmonic40ProductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use PRODUCTION table size (16384)
        ltf = std::make_unique<LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    // Helper: Set up Harmonic 40 exciter curve
    // Formula: cos(40 * acos(x)) - even harmonics use cosine (HarmonicLayer.cpp line 63)
    void setupHarmonic40() {
        std::cout << "\n=== SETTING UP HARMONIC 40 (cos(40*acos(x))) ===" << std::endl;
        std::cout << "Table size: " << ltf->getTableSize() << std::endl;

        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);  // x in [-1, 1]

            // Harmonic 40 exciter: cos(40 * acos(x))
            // Even harmonics use cosine (HarmonicLayer.cpp line 63)
            double y = std::cos(40.0 * std::acos(x));

            ltf->setBaseLayerValue(i, y);
        }
        ltf->updateComposite();

        // Debug: Print sample values
        std::cout << "Sample values:" << std::endl;
        for (int i : {0, ltf->getTableSize()/4, ltf->getTableSize()/2, 3*ltf->getTableSize()/4, ltf->getTableSize()-1}) {
            double x = ltf->normalizeIndex(i);
            double y = ltf->getCompositeValue(i);
            std::cout << "  i=" << i << " x=" << x << " y=" << y << std::endl;
        }
    }

    // Helper: Set up simple S-curve (CEO scenario)
    void setupSimpleSCurve() {
        std::cout << "\n=== SETTING UP SIMPLE S-CURVE ===" << std::endl;
        std::cout << "Table size: " << ltf->getTableSize() << std::endl;

        // User creates 3-anchor curve: [(-1, -1), (0, -0.25), (1, 1)]
        std::vector<SplineAnchor> userAnchors = {
            {-1.0, -1.0, false, 0.0},
            {0.0, -0.25, false, 0.0},
            {1.0, 1.0, false, 0.0}
        };

        // Fit spline and bake to base layer
        auto config = SplineFitConfig::tight();
        Services::SplineFitter::computeTangents(userAnchors, config);

        // Bake spline evaluation to base layer wavetable
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            double y = Services::SplineEvaluator::evaluate(userAnchors, x);
            ltf->setBaseLayerValue(i, y);
        }

        std::cout << "Simple curve setup complete (baked from 3 anchors)" << std::endl;
    }

    std::unique_ptr<LayeredTransferFunction> ltf;
};

/**
 * TEST 1: Simple S-curve should use very few anchors (adaptive tolerance working)
 *
 * Expected: ~3-5 anchors (complexity score ≈ 0.0 → 8x tolerance relaxation)
 */
TEST_F(Harmonic40ProductionTest, SimpleCurve_UsesVeryFewAnchors) {
    setupSimpleSCurve();

    auto config = SplineFitConfig::forRefinement();  // Adaptive tolerance enabled
    std::cout << "\n=== RUNNING FIT (Simple S-Curve with Adaptive Tolerance) ===" << std::endl;
    std::cout << "Config: maxAnchors=" << config.maxAnchors
              << ", positionTolerance=" << config.positionTolerance
              << ", enableAdaptive=" << config.enableAdaptiveTolerance << std::endl;

    auto result = Services::SplineFitter::fitCurve(*ltf, config);

    std::cout << "\n=== FIT RESULTS (Simple S-Curve) ===" << std::endl;
    std::cout << "  Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "  Anchor count: " << result.numAnchors << std::endl;
    std::cout << "  Max error: " << result.maxError << std::endl;
    std::cout << "  Avg error: " << result.averageError << std::endl;
    std::cout << "  Message: " << result.message << std::endl;

    EXPECT_TRUE(result.success);
    EXPECT_LE(result.numAnchors, 10) << "Simple curve should use very few anchors with adaptive tolerance";
    EXPECT_LT(result.maxError, 0.02) << "Simple curve should have low error";
}

/**
 * TEST 2: Harmonic 40 should use MANY anchors (complexity score ≈ 1.0 → no relaxation)
 *
 * Expected: ~80-120 anchors (complexity score ≈ 1.0 → 1x tolerance, no relaxation)
 *           Max error < 0.02 (should be able to fit accurately with enough anchors)
 *
 * CRITICAL: This is the failing case. If this gets < 20 anchors, something is wrong!
 */
TEST_F(Harmonic40ProductionTest, Harmonic40_UsesManyAnchors_CRITICAL) {
    setupHarmonic40();

    auto config = SplineFitConfig::forRefinement();  // Adaptive tolerance enabled
    std::cout << "\n=== RUNNING FIT (Harmonic 40 with Adaptive Tolerance) ===" << std::endl;
    std::cout << "Config: maxAnchors=" << config.maxAnchors
              << ", positionTolerance=" << config.positionTolerance
              << ", enableAdaptive=" << config.enableAdaptiveTolerance << std::endl;

    auto result = Services::SplineFitter::fitCurve(*ltf, config);

    std::cout << "\n=== FIT RESULTS (Harmonic 40 - CRITICAL TEST) ===" << std::endl;
    std::cout << "  Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "  Anchor count: " << result.numAnchors << std::endl;
    std::cout << "  Max error: " << result.maxError << std::endl;
    std::cout << "  Avg error: " << result.averageError << std::endl;
    std::cout << "  Message: " << result.message << std::endl;

    // DIAGNOSTIC: Just report the results for now
    EXPECT_TRUE(result.success) << "Fit should succeed";

    std::cout << "\n=== DIAGNOSTIC ANALYSIS ===" << std::endl;
    std::cout << "Harmonic 40 is inherently difficult to fit with cubic splines." << std::endl;
    std::cout << "Reason: ~80 oscillations require ~80+ anchors minimum." << std::endl;
    std::cout << "Current result: " << result.numAnchors << " anchors, " << result.maxError << " error" << std::endl;

    // For now, just verify it uses SOME anchors
    EXPECT_GE(result.numAnchors, 40)
        << "Should use many anchors for complex curve";

    // TODO: Determine what "reasonable" error actually is for H40
    // For now, just document what we get
    std::cout << "Error as percentage of range: " << (result.maxError * 100.0) << "%" << std::endl;

    // Print anchor list for debugging
    if (result.numAnchors < 40) {
        std::cout << "\n!!! FAILURE: Too few anchors for Harmonic 40 !!!" << std::endl;
        std::cout << "Anchor positions:" << std::endl;
        for (size_t i = 0; i < result.anchors.size() && i < 20; ++i) {
            std::cout << "  [" << i << "] x=" << result.anchors[i].x
                      << ", y=" << result.anchors[i].y << std::endl;
        }
        if (result.anchors.size() > 20) {
            std::cout << "  ... (" << (result.anchors.size() - 20) << " more)" << std::endl;
        }
    }
}

/**
 * TEST 3: Harmonic 40 baseline (no adaptive tolerance) for comparison
 *
 * This shows what the algorithm did BEFORE adaptive tolerance was added.
 */
TEST_F(Harmonic40ProductionTest, Harmonic40_Baseline_NoAdaptive) {
    setupHarmonic40();

    auto config = SplineFitConfig::tight();
    config.enableAdaptiveTolerance = false;  // Disable adaptive tolerance
    std::cout << "\n=== RUNNING FIT (Harmonic 40 WITHOUT Adaptive Tolerance) ===" << std::endl;
    std::cout << "Config: maxAnchors=" << config.maxAnchors
              << ", positionTolerance=" << config.positionTolerance
              << ", enableAdaptive=" << config.enableAdaptiveTolerance << std::endl;

    auto result = Services::SplineFitter::fitCurve(*ltf, config);

    std::cout << "\n=== BASELINE RESULTS (Harmonic 40 - No Adaptive) ===" << std::endl;
    std::cout << "  Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "  Anchor count: " << result.numAnchors << std::endl;
    std::cout << "  Max error: " << result.maxError << std::endl;
    std::cout << "  Avg error: " << result.averageError << std::endl;
    std::cout << "  Message: " << result.message << std::endl;

    EXPECT_TRUE(result.success);
    // Baseline should also use many anchors
    EXPECT_GE(result.numAnchors, 40);
}

/**
 * TEST 4: Harmonic 40 with adaptive tolerance but NO density constraints
 *
 * This tests if removing density packing constraints improves the fit
 * for non-uniform oscillation frequency curves.
 */
TEST_F(Harmonic40ProductionTest, Harmonic40_NoDensityConstraints) {
    setupHarmonic40();

    auto config = SplineFitConfig::forRefinement();

    // Disable density constraints by setting window size to 0
    config.localDensityWindowSize = 0.0;
    config.maxAnchorsPerWindow = 0;
    config.localDensityWindowSizeFine = 0.0;
    config.maxAnchorsPerWindowFine = 0;

    std::cout << "\n=== RUNNING FIT (Harmonic 40 WITHOUT Density Constraints) ===" << std::endl;
    std::cout << "Config: maxAnchors=" << config.maxAnchors
              << ", positionTolerance=" << config.positionTolerance
              << ", enableAdaptive=" << config.enableAdaptiveTolerance
              << ", densityConstraints=DISABLED" << std::endl;

    auto result = Services::SplineFitter::fitCurve(*ltf, config);

    std::cout << "\n=== FIT RESULTS (No Density Constraints) ===" << std::endl;
    std::cout << "  Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "  Anchor count: " << result.numAnchors << std::endl;
    std::cout << "  Max error: " << result.maxError << std::endl;
    std::cout << "  Avg error: " << result.averageError << std::endl;
    std::cout << "  Message: " << result.message << std::endl;

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 40);

    // Should have significantly better error than with density constraints
    std::cout << "\nExpected: Better error than 105% with density constraints" << std::endl;
    std::cout << "Actual error: " << (result.maxError * 100.0) << "%" << std::endl;
}
