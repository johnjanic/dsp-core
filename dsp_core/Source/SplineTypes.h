#pragma once
#include <vector>
#include <juce_core/juce_core.h>

namespace dsp_core {

// Tangent computation algorithms for spline fitting
enum class TangentAlgorithm {
    PCHIP,              // Piecewise Cubic Hermite Interpolating Polynomial (default)
    FritschCarlson,     // Monotone-preserving variant
    Akima,              // Local weighted average
    FiniteDifference    // Simple baseline (for comparison)
};

// Control point with optional tangent override
struct SplineAnchor {
    double x = 0.0;  // Position in [-1, 1]
    double y = 0.0;  // Value in [-1, 1]
    bool hasCustomTangent = false;
    double tangent = 0.0;  // m_i in PCHIP notation

    bool operator==(const SplineAnchor&) const = default;
};

// Result of curve fitting operation
struct SplineFitResult {
    bool success = false;
    std::vector<SplineAnchor> anchors;
    double maxError = 0.0;  // Peak absolute error |y - ŷ|
    int numAnchors = 0;
    juce::String message;  // User-facing feedback

    // Statistics for user feedback
    double averageError = 0.0;
    double maxDerivativeError = 0.0;
};

/**
 * Local Density Constraint:
 *   Prevents anchor clustering in small regions (e.g., scribbles consuming entire budget)
 *   while allowing high total anchor counts for globally complex curves (e.g., Harmonic 40).
 *
 *   Example: windowSize=0.10, maxAnchorsPerWindow=8
 *     - 10 windows × 8 anchors/window = 80 anchors max (globally distributed)
 *     - Scribble in 12% region = 1.2 windows × 8 = ~10 anchors (locally limited)
 */
// Configuration for fitting algorithm
struct SplineFitConfig {
    // RDP tolerance (hybrid error metric)
    double positionTolerance = 0.01;      // α in professor's notation (0.002-0.01)
    double derivativeTolerance = 0.02;    // β in professor's notation (0.02-0.05)

    // Refinement parameters
    int maxAnchors = 128;                 // K_max - Increased to allow complex curves (was 64)
    bool enableRefinement = true;

    // Local density constraint (prevents anchor clustering in small regions)
    // Enforces: "No more than maxAnchorsPerWindow anchors within any windowSize sliding window"
    double localDensityWindowSize = 0.10;     // Window size as fraction of domain (tunable: 0.08-0.12)
    int maxAnchorsPerWindow = 8;              // Max anchors within any window (tunable: 6-10)

    // Monotonicity enforcement
    bool enforceMonotonicity = true;

    // Slope bounds (for anti-aliasing)
    double minSlope = -8.0;               // m_min
    double maxSlope = 8.0;                // m_max

    // Endpoint behavior
    bool pinEndpoints = true;             // Force (-1,-1) and (1,1)

    // Tangent computation algorithm
    // NOTE: Fritsch-Carlson chosen for no-overshoot guarantee (simplifies UI anchor placement)
    // See docs/architecture/spline-algorithm-decision.md for full analysis
    TangentAlgorithm tangentAlgorithm = TangentAlgorithm::FritschCarlson;

    // Presets
    static SplineFitConfig tight() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.002;
        cfg.derivativeTolerance = 0.05;
        cfg.maxAnchors = 128;  // Changed from 64
        cfg.localDensityWindowSize = 0.10;
        cfg.maxAnchorsPerWindow = 8;
        cfg.tangentAlgorithm = TangentAlgorithm::FritschCarlson;
        return cfg;
    }

    static SplineFitConfig smooth() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.01;
        cfg.derivativeTolerance = 0.02;
        cfg.maxAnchors = 64;  // Moderate complexity
        cfg.localDensityWindowSize = 0.10;
        cfg.maxAnchorsPerWindow = 8;
        cfg.tangentAlgorithm = TangentAlgorithm::FritschCarlson;
        return cfg;
    }

    static SplineFitConfig monotonePreserving() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.001;
        cfg.derivativeTolerance = 0.02;
        cfg.maxAnchors = 32;
        cfg.tangentAlgorithm = TangentAlgorithm::FritschCarlson;
        return cfg;
    }
};

} // namespace dsp_core
