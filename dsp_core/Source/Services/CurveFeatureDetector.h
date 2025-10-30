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
     * @return Feature indices (table indices, not normalized coordinates)
     */
    static FeatureResult detectFeatures(const LayeredTransferFunction& ltf, int maxMandatoryAnchors = 0);

private:
    CurveFeatureDetector() = delete;  // Pure static service

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
};

} // namespace Services
} // namespace dsp_core
