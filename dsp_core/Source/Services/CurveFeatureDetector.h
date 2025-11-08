#pragma once
#include "../LayeredTransferFunction.h"
#include <vector>

namespace dsp_core {
namespace Services {

/**
 * CurveFeatureDetector - Pure service for geometric feature detection in curves
 *
 * Detects critical geometric features (local extrema, inflection points) in
 * transfer function curves that require anchors for accurate spline fitting.
 *
 * Purpose:
 *   Eliminate ripple/oscillation artifacts by ensuring spline anchors are placed
 *   at all geometric features. A cubic segment can represent at most 1 inflection
 *   point and 2 local extrema. If data has more features between anchors, we're
 *   undersampling (like violating Nyquist frequency), which causes ripple.
 *
 * Algorithm:
 *   1. Detect local extrema (dy/dx sign changes) - peaks and valleys
 *   2. Detect inflection points (d²y/dx² sign changes) - curvature reversals
 *   3. Merge features into mandatory anchor list (always include endpoints)
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class CurveFeatureDetector {
public:
    struct FeatureResult {
        std::vector<int> localExtrema;       // peaks and valleys (dy/dx sign changes)
        std::vector<int> inflectionPoints;   // d²y/dx² sign changes
        std::vector<int> mandatoryAnchors;   // merged + sorted (always include these)
    };

    /**
     * Detect all geometric features requiring spline anchors
     *
     * @param ltf Input transfer function (base layer)
     * @param maxMandatoryAnchors Maximum number of mandatory feature anchors (0 = unlimited)
     *                            If limited, keeps most significant features by amplitude/curvature
     * @param localDensityWindowSize Coarse window size as fraction of domain (0.0 = disabled)
     * @param maxAnchorsPerWindow Max anchors within coarse window (0 = disabled)
     * @param localDensityWindowSizeFine Fine window size (0.0 = disabled, catches pixel-level clustering)
     * @param maxAnchorsPerWindowFine Max anchors within fine window (0 = disabled)
     * @return Feature indices (table indices, not normalized coordinates)
     */
    static FeatureResult detectFeatures(
        const LayeredTransferFunction& ltf,
        int maxMandatoryAnchors = 0,
        double localDensityWindowSize = 0.0,        // Coarse: 0.0 = disabled (backward compatible)
        int maxAnchorsPerWindow = 0,                 // Coarse: 0 = disabled
        double localDensityWindowSizeFine = 0.0,     // Fine: 0.0 = disabled
        int maxAnchorsPerWindowFine = 0              // Fine: 0 = disabled
    );

private:
    CurveFeatureDetector() = delete;  // Pure static service

    // Internal feature representation for prioritization
    struct Feature {
        int index;
        double significance;  // Global significance (amplitude or curvature)
        double prominence;    // Prominence-based significance (local context)
        bool isExtremum;      // true = extremum, false = inflection point
    };

    /**
     * Compute topographic prominence for a peak/valley
     *
     * Prominence = height above the higher of the two nearest valleys (for peaks)
     *            = depth below the lower of the two nearest peaks (for valleys)
     *
     * This measures how much a feature "stands out" from its surroundings,
     * independent of global amplitude. A small peak in a quiet region can have
     * high prominence, while a large peak in a loud region may have low prominence.
     *
     * @param feature The feature to evaluate
     * @param allFeatures All detected features (for finding reference points)
     * @param ltf Transfer function (for y-values)
     * @return Prominence value [0.0, 2.0] (normalized to range)
     */
    static double computeProminence(
        const Feature& feature,
        const std::vector<Feature>& allFeatures,
        const LayeredTransferFunction& ltf
    );

    /**
     * Classify feature as Mandatory, Significant, or Minor
     *
     * - Mandatory: Always include (endpoints, global extrema, sharp inflections)
     * - Significant: Include if budget allows (prominent local features)
     * - Minor: Skip for sparseness (small wiggles, scribble noise)
     */
    enum class FeatureTier {
        Mandatory,   // Must include for correctness
        Significant, // Include if budget allows
        Minor        // Skip for sparseness
    };

    static FeatureTier classifyFeature(
        const Feature& feature,
        const std::vector<Feature>& allFeatures,
        const LayeredTransferFunction& ltf,
        int tableSize
    );

    /**
     * Estimate first derivative at index using central difference
     * @param ltf Input transfer function
     * @param idx Table index
     * @return Estimated dy/dx
     */
    static double estimateDerivative(const LayeredTransferFunction& ltf, int idx);

    /**
     * Estimate second derivative at index using finite difference
     * @param ltf Input transfer function
     * @param idx Table index
     * @return Estimated d²y/dx²
     */
    static double estimateSecondDerivative(const LayeredTransferFunction& ltf, int idx);

    /**
     * Count anchors within sliding window centered at candidateIdx
     *
     * @param candidateIdx Table index of feature being considered
     * @param existingAnchors Already-selected anchor indices
     * @param tableSize Total table size (e.g., 256)
     * @param windowSizeFraction Window width as fraction of domain (e.g., 0.10 = 10%)
     * @return Number of existing anchors within ±(windowSize/2) of candidate
     */
    static int countAnchorsInWindow(
        int candidateIdx,
        const std::vector<int>& existingAnchors,
        int tableSize,
        double windowSizeFraction
    );
};

} // namespace Services
} // namespace dsp_core
