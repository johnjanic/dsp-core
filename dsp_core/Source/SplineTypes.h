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
 * Configuration for spline fitting algorithm.
 *
 * ADAPTIVE TOLERANCE SYSTEM (Phase 1 - Nov 2025):
 * - enableAdaptiveTolerance: Scales tolerance based on curve complexity
 * - toleranceScaleFactor: Multiplier for simple curves (8.0 = 8x relaxation)
 * - minRelativeImprovement: Stop if improvement < threshold (prevents noise chasing)
 *
 * MODE-AWARE PRESETS:
 * - forExploration(): Aggressive sparsity (scale=10.0, improvement=10%)
 * - forRefinement(): Balanced default (scale=8.0, improvement=5%)
 * - forConversion(): Preserve complexity (scale=5.0, improvement=3%)
 *
 * BACKWARD COMPATIBILITY:
 * - enableAdaptiveTolerance defaults to FALSE
 * - Existing presets (tight/smooth/monotonePreserving) unchanged
 * - Opt-in via forRefinement() or enableAdaptiveTolerance=true
 *
 * Local Density Constraint:
 *   Prevents anchor clustering in small regions (e.g., scribbles consuming entire budget)
 *   while allowing high total anchor counts for globally complex curves (e.g., Harmonic 40).
 *
 *   Example: windowSize=0.10, maxAnchorsPerWindow=8
 *     - 10 windows × 8 anchors/window = 80 anchors max (globally distributed)
 *     - Scribble in 12% region = 1.2 windows × 8 = ~10 anchors (locally limited)
 */
struct SplineFitConfig {
    // RDP tolerance (hybrid error metric)
    double positionTolerance = 0.01;      // α in professor's notation (0.002-0.01)
    double derivativeTolerance = 0.02;    // β in professor's notation (0.02-0.05)

    // Refinement parameters
    int maxAnchors = 128;                 // K_max - Increased to allow complex curves (was 64)
    bool enableRefinement = true;

    // Two-tier local density constraint (prevents anchor clustering at multiple scales)
    // Coarse constraint: Prevents regional hogging (e.g., scribble consuming 50% of budget)
    // VALIDATED: 0.10, 8 optimal (see docs/feature-plans/curve-fitting-tuning-results.md)
    double localDensityWindowSize = 0.10;     // Window size as fraction of domain (tunable: 0.08-0.12)
    int maxAnchorsPerWindow = 8;              // Max anchors within any window (tunable: 6-10)

    // Fine constraint: Prevents pixel-level clustering (e.g., 3 anchors within 20px)
    // VALIDATED: 0.02, 2 optimal (complements coarse constraint)
    // Set to 0.0 to disable (disabled by default until further validation)
    double localDensityWindowSizeFine = 0.0;  // Fine window: ~20px at 1000px width (tunable: 0.015-0.03)
    int maxAnchorsPerWindowFine = 0;          // Max 2 anchors within fine window (tunable: 1-3)

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

    // ===== NEW: Adaptive Tolerance System (Phase 1 - Nov 2025) =====

    // Enable/disable adaptive tolerance (opt-in for backward compatibility)
    bool enableAdaptiveTolerance = false;

    // Tolerance scaling factor (multiplier for simple curves)
    // Simple curves: tolerance = base × scaleFactor^(1-complexity)
    // Complex curves: tolerance = base × 1.0
    double toleranceScaleFactor = 8.0;

    // Diminishing returns detection
    bool enableRelativeImprovementCheck = false;

    // Minimum relative improvement per iteration (5% default)
    // relativeImprovement = (prevError - curError) / prevError
    double minRelativeImprovement = 0.05;

    // Max consecutive iterations below improvement threshold
    int maxSlowProgressIterations = 3;

    // ===== END NEW =====

    // Presets
    static SplineFitConfig tight() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.002;
        cfg.derivativeTolerance = 0.05;
        cfg.maxAnchors = 128;  // Changed from 64
        cfg.localDensityWindowSize = 0.10;
        cfg.maxAnchorsPerWindow = 8;
        cfg.localDensityWindowSizeFine = 0.02;  // Enable fine constraint
        cfg.maxAnchorsPerWindowFine = 2;
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
        cfg.localDensityWindowSizeFine = 0.02;  // Enable fine constraint
        cfg.maxAnchorsPerWindowFine = 2;
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

    // ===== NEW: Mode-Aware Presets (Phase 1 - Nov 2025) =====

    // Exploration mode: Aggressive sparsity for quick iteration
    static SplineFitConfig forExploration() {
        auto cfg = tight();  // Start with tight base
        cfg.enableAdaptiveTolerance = true;
        cfg.enableRelativeImprovementCheck = true;
        cfg.toleranceScaleFactor = 10.0;  // Very aggressive
        cfg.minRelativeImprovement = 0.10;  // Stop early (10%)
        cfg.maxSlowProgressIterations = 3;
        return cfg;
    }

    // Refinement mode: Balanced for user editing (DEFAULT)
    static SplineFitConfig forRefinement() {
        auto cfg = tight();
        cfg.enableAdaptiveTolerance = true;
        cfg.enableRelativeImprovementCheck = true;
        cfg.toleranceScaleFactor = 8.0;  // Balanced
        cfg.minRelativeImprovement = 0.05;  // Standard (5%)
        cfg.maxSlowProgressIterations = 3;
        return cfg;
    }

    // Conversion mode: Preserve complexity when baking harmonics
    static SplineFitConfig forConversion() {
        auto cfg = tight();
        cfg.enableAdaptiveTolerance = true;
        cfg.enableRelativeImprovementCheck = true;
        cfg.toleranceScaleFactor = 5.0;  // Conservative
        cfg.minRelativeImprovement = 0.03;  // Tight convergence (3%)
        cfg.maxSlowProgressIterations = 5;  // More patient
        return cfg;
    }

    // Alias: Default adaptive preset (maps to forRefinement)
    static SplineFitConfig adaptiveTight() {
        return forRefinement();
    }

    // ===== END NEW =====

    // Optional: Validation method
    bool validate() const {
        if (toleranceScaleFactor < 1.0 || toleranceScaleFactor > 20.0) {
            DBG("Invalid toleranceScaleFactor: " << toleranceScaleFactor
                << " (must be 1.0-20.0)");
            return false;
        }
        if (minRelativeImprovement < 0.0 || minRelativeImprovement > 1.0) {
            DBG("Invalid minRelativeImprovement: " << minRelativeImprovement
                << " (must be 0.0-1.0)");
            return false;
        }
        if (maxSlowProgressIterations < 1 || maxSlowProgressIterations > 20) {
            DBG("Invalid maxSlowProgressIterations: " << maxSlowProgressIterations
                << " (must be 1-20)");
            return false;
        }
        return true;
    }
};

} // namespace dsp_core
