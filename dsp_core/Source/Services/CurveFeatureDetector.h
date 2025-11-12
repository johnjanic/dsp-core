#pragma once
#include <vector>

namespace dsp_core {

// Forward declaration to break circular dependency
class LayeredTransferFunction;

/**
 * Configuration for feature detection and significance filtering
 */
struct FeatureDetectionConfig {
    /**
     * Significance threshold as percentage of vertical range (0.0-1.0)
     * Extrema with local amplitude change less than this threshold are discarded
     * Default: 0.001 (0.1% of vertical range)
     */
    double significanceThreshold;

    /**
     * Maximum number of features to detect (0 = unlimited)
     * If limited, keeps most significant features by amplitude/curvature
     */
    int maxFeatures;

    /**
     * First derivative noise floor - ignore extrema with |dy/dx| < threshold
     * Lower = more sensitive (detects subtle peaks)
     * Higher = more conservative (ignores noise)
     * Range: 1e-7 to 1e-5, Default: 1e-6
     */
    double derivativeThreshold;

    /**
     * Second derivative noise floor - ignore inflections with |d²y/dx²| < threshold
     * Lower = more sensitive (detects subtle curvature changes)
     * Higher = more conservative (ignores numerical noise)
     * Range: 1e-5 to 1e-3, Default: 1e-4
     */
    double secondDerivativeThreshold;

    /**
     * Budget ratio for extrema vs inflection points
     * When maxFeatures limit is hit, prioritize extrema over inflections
     * Value = fraction of budget for extrema (remainder goes to inflections)
     * Range: 0.5 (equal priority) to 1.0 (extrema only), Default: 0.8
     */
    double extremaInflectionRatio;

    /**
     * Default constructor - uses sensible defaults
     * Note: Very low significance threshold (0.1%) to preserve all real features
     * Increase threshold (e.g., 2-5%) for noise filtering
     */
    FeatureDetectionConfig()
        : significanceThreshold(0.001)
        , maxFeatures(100)
        , derivativeThreshold(1e-6)
        , secondDerivativeThreshold(1e-4)
        , extremaInflectionRatio(0.8)
    {}

    bool operator==(const FeatureDetectionConfig&) const = default;
};

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
     * @param config Configuration for detection and filtering (optional)
     * @return Feature indices (table indices, not normalized coordinates)
     */
    static FeatureResult detectFeatures(const LayeredTransferFunction& ltf,
                                       const FeatureDetectionConfig& config = FeatureDetectionConfig{});

    /**
     * Legacy overload for backward compatibility
     * @deprecated Use version with FeatureDetectionConfig instead
     */
    static FeatureResult detectFeatures(const LayeredTransferFunction& ltf, int maxMandatoryAnchors);

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
