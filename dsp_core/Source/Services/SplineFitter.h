#pragma once
#include "../SplineTypes.h"
#include "../LayeredTransferFunction.h"
#include "CurveFeatureDetector.h"

namespace dsp_core {
namespace Services {

/**
 * SplineFitter - Pure service for spline curve fitting
 *
 * Converts painted baseLayer curve to control points + spline.
 *
 * Algorithm (3 steps):
 *   1. Sample & Sanitize: baseLayer → clean polyline
 *   2. Uniform Anchor Placement: polyline → control points
 *   3. Tangent Computation: compute spline tangents (PCHIP, Akima, etc.)
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

    // Legacy API: compute PCHIP tangents specifically
    // Deprecated: Use computeTangents() with config.tangentAlgorithm = TangentAlgorithm::PCHIP
    [[deprecated("Use computeTangents() with config.tangentAlgorithm instead")]]
    static void computePCHIPTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config = SplineFitConfig::smooth()
    );

private:
    // Step 1: Sample curve
    struct Sample { double x, y; };
    static std::vector<Sample> sampleAndSanitize(
        const LayeredTransferFunction& ltf,
        const SplineFitConfig& config
    );

    // Step 2: Tangent computation algorithms
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

    // Simple uniform anchor placement
    static std::vector<SplineAnchor> greedySplineFit(
        const std::vector<Sample>& samples,
        const SplineFitConfig& config,
        const LayeredTransferFunction* ltf = nullptr,
        const std::vector<int>& mandatoryAnchorIndices = {},
        const CurveFeatureDetector::FeatureResult* features = nullptr
    );

    // Uniform sampling fallback (legacy)
    static std::vector<SplineAnchor> greedySplineFitUniform(
        const std::vector<Sample>& samples,
        const SplineFitConfig& config
    );

    SplineFitter() = delete;  // Pure static utility
};

} // namespace Services
} // namespace dsp_core
