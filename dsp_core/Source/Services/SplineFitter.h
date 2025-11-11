#pragma once
#include "../SplineTypes.h"
#include "../LayeredTransferFunction.h"

namespace dsp_core {
namespace Services {

/**
 * SplineFitter - Pure service for spline curve fitting
 *
 * Converts painted baseLayer curve to control points + cubic Hermite spline
 * using feature-based initialization and greedy error-driven refinement.
 *
 * Algorithm (5 stages):
 *   1. Feature Detection: identify extrema and inflection points
 *   2. Sample & Sanitize: baseLayer → clean polyline
 *   3. Greedy Fitting: feature anchors + error-driven refinement
 *   4. Tangent Computation: compute Fritsch-Carlson monotone tangents
 *   5. Error Analysis: compute fit quality metrics
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class SplineFitter {
public:
    // Main API: Fit painted curve to spline anchors
    static SplineFitResult fitCurve(
        const LayeredTransferFunction& ltf,
        const SplineFitConfig& config = SplineFitConfig::smooth()
    );

    // Tangent computation (exposed for manual anchor manipulation)
    // Recomputes tangents for anchors after position changes using configured algorithm
    static void computeTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config = SplineFitConfig::smooth()
    );

    // Zero-crossing analysis (exposed for testing)
    struct ZeroCrossingInfo {
        bool baseCurveHasZeroCrossing = false;
        double baseYAtZero = 0.0;
        double fittedYAtZero = 0.0;
        double drift = 0.0;
    };

    static ZeroCrossingInfo analyzeZeroCrossing(
        const LayeredTransferFunction& ltf,
        const std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

private:
    // Step 1: Sample & sanitize
    struct Sample { double x, y; };
    static std::vector<Sample> sampleAndSanitize(
        const LayeredTransferFunction& ltf,
        const SplineFitConfig& config
    );

    // Sub-steps of sanitize
    static void sortByX(std::vector<Sample>& samples);
    static void deduplicateNearVerticals(std::vector<Sample>& samples);
    static void enforceMonotonicity(std::vector<Sample>& samples);
    static void clampToRange(std::vector<Sample>& samples);

    // Tangent computation algorithms
    static void computePCHIPTangentsImpl(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

    static void computeFritschCarlsonTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

    static void computeAkimaTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

    static void computeFiniteDifferenceTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

    // Tangent computation helpers
    static double harmonicMean(double a, double b, double wa, double wb);

    // Greedy spline fitting (replaces RDP + refinement)
    // Now uses feature-based anchor placement: starts with mandatory feature anchors
    static std::vector<SplineAnchor> greedySplineFit(
        const std::vector<Sample>& samples,
        const SplineFitConfig& config,
        const LayeredTransferFunction* ltf = nullptr,
        const std::vector<int>& mandatoryAnchorIndices = {}
    );

    // Find sample with worst fit error
    struct WorstFitResult {
        size_t sampleIndex;
        double maxError;
    };

    static WorstFitResult findWorstFitSample(
        const std::vector<Sample>& samples,
        const std::vector<SplineAnchor>& anchors
    );

    // Anchor pruning (optional post-processing)
    static void pruneRedundantAnchors(
        std::vector<SplineAnchor>& anchors,
        const std::vector<Sample>& samples,
        double pruningTolerance,
        const SplineFitConfig& config
    );

    SplineFitter() = delete;  // Pure static utility
};

} // namespace Services
} // namespace dsp_core
