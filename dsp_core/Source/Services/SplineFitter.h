#pragma once
#include "../SplineTypes.h"
#include "../LayeredTransferFunction.h"

namespace dsp_core {
namespace Services {

/**
 * SplineFitter - Pure service for PCHIP spline curve fitting
 *
 * Converts painted baseLayer curve to control points + PCHIP spline
 * using Ramer-Douglas-Peucker simplification and Fritsch-Carlson tangents.
 *
 * Algorithm (4 steps):
 *   1. Sample & Sanitize: baseLayer → clean polyline
 *   2. RDP Simplification: polyline → control points
 *   3. PCHIP Tangents: compute Fritsch-Carlson monotone tangents
 *   4. Error Analysis: compute fit quality metrics
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

    // Step 2: Ramer-Douglas-Peucker simplification
    static std::vector<SplineAnchor> ramerDouglasPeucker(
        const std::vector<Sample>& samples,
        const SplineFitConfig& config
    );

    static void rdpRecursive(
        const std::vector<Sample>& samples,
        size_t startIdx,
        size_t endIdx,
        const SplineFitConfig& config,
        std::vector<bool>& keep
    );

    static double computeHybridError(
        const Sample& point,
        const Sample& lineStart,
        const Sample& lineEnd,
        double alpha,
        double beta
    );

    static double estimateDerivative(
        const std::vector<Sample>& samples,
        size_t index
    );

    // Step 3: PCHIP tangent computation
    static void computePCHIPTangents(
        std::vector<SplineAnchor>& anchors,
        const SplineFitConfig& config
    );

    static double harmonicMean(double a, double b, double wa, double wb);

    SplineFitter() = delete;  // Pure static utility
};

} // namespace Services
} // namespace dsp_core
