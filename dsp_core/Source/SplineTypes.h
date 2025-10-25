#pragma once
#include <vector>
#include <juce_core/juce_core.h>

namespace dsp_core {

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

// Configuration for fitting algorithm
struct SplineFitConfig {
    // RDP tolerance (hybrid error metric)
    double positionTolerance = 0.01;      // α in professor's notation (0.002-0.01)
    double derivativeTolerance = 0.02;    // β in professor's notation (0.02-0.05)

    // Refinement parameters
    int maxAnchors = 64;                  // K_max
    bool enableRefinement = true;

    // Monotonicity enforcement
    bool enforceMonotonicity = true;

    // Slope bounds (for anti-aliasing)
    double minSlope = -8.0;               // m_min
    double maxSlope = 8.0;                // m_max

    // Endpoint behavior
    bool pinEndpoints = true;             // Force (-1,-1) and (1,1)

    // Presets
    static SplineFitConfig tight() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.002;
        cfg.derivativeTolerance = 0.05;
        cfg.maxAnchors = 64;
        return cfg;
    }

    static SplineFitConfig smooth() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.01;
        cfg.derivativeTolerance = 0.02;
        cfg.maxAnchors = 24;
        return cfg;
    }
};

} // namespace dsp_core
