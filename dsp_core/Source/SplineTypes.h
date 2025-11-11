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

// Symmetry mode for anchor placement
enum class SymmetryMode {
    Auto,    // Auto-detect symmetry, enable if score > threshold
    Always,  // Force symmetric fitting regardless of detection
    Never    // Disable symmetric fitting (original behavior)
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
 * Configuration for spline curve fitting algorithm.
 *
 * The fitting algorithm uses adaptive tolerance to prevent "anchor creeping" - the problem
 * where refitting curves produces exponentially more anchors on each cycle.
 *
 * Key behavioral characteristics:
 * - Adaptive tolerance increases with anchor density to prevent over-fitting
 * - Backtranslation stable: refit curves converge to minimal anchor count
 * - Anchor counts scale with geometric complexity (matches Chebyshev extrema)
 * - Typical results: H1=2, H2=2, H3=3, H5=5, H10=10 anchors
 *
 * See docs/architecture/spline-curve-fitting.md for detailed algorithm documentation.
 */
struct SplineFitConfig {
    /**
     * Position tolerance - baseline error threshold for curve fitting.
     *
     * This serves as both:
     * 1. Minimum floor for adaptive tolerance (early iterations use this value)
     * 2. Basis for computing adaptive tolerance: baselineTolerance = positionTolerance / verticalRange
     *
     * The adaptive tolerance formula:
     *   tolerance = baselineTolerance × (1 + anchorRatio × 2.0)
     *   where anchorRatio = currentAnchors / maxAnchors
     *
     * Effect on fitting:
     * - Lower values (0.001-0.002): Tighter fit, more anchors, higher quality
     * - Higher values (0.01-0.02): Looser fit, fewer anchors, smoother curves
     *
     * Range: 0.001 (tight) to 0.02 (smooth)
     * Recommended: 0.01 for general use, 0.002 for high-precision work
     */
    double positionTolerance = 0.01;

    /**
     * Derivative tolerance - unused in adaptive algorithm (legacy parameter).
     *
     * Originally used for RDP simplification. Now kept for backward compatibility
     * but not actively used in current greedy fitting algorithm.
     */
    double derivativeTolerance = 0.02;

    /**
     * Maximum anchor budget - hard limit on anchor count.
     *
     * Interacts with adaptive tolerance:
     * - As anchor count approaches maxAnchors, tolerance increases to prevent over-fitting
     * - Prevents pathological cases where algorithm adds unlimited anchors
     * - Should be set based on UI performance requirements (more anchors = more drag latency)
     *
     * Effect on fitting:
     * - Lower values (16-24): Faster UI, fewer control points, may under-fit complex curves
     * - Higher values (64-128): More precision, can capture very complex curves, slower UI
     *
     * Range: 16 (minimal) to 128 (high precision)
     * Recommended: 24 for interactive editing, 64 for offline processing
     */
    int maxAnchors = 64;

    /**
     * Enable greedy refinement phase.
     *
     * When true: Uses feature detection + error-driven refinement (default, recommended)
     * When false: Uses only feature detection (fast but lower quality)
     *
     * Should always be true unless benchmarking or debugging.
     */
    bool enableRefinement = true;

    /**
     * Enforce monotonicity - prevent y-backtracking.
     *
     * When true: Transfer function is guaranteed to be monotonically increasing
     * When false: Allows folding distortion effects (y can decrease as x increases)
     *
     * For audio waveshaping:
     * - True: Standard waveshaping (no phase inversion artifacts)
     * - False: Creative effects (folding distortion, wavefolding)
     *
     * Recommended: true for production, false for experimental effects
     */
    bool enforceMonotonicity = true;

    /**
     * Slope bounds for anti-aliasing.
     *
     * Clamps tangent values to prevent extreme slopes that could cause aliasing
     * in audio processing. Range of [-8, 8] allows steep curves while preventing
     * numerical instability.
     *
     * Formula: tangent = clamp(computedTangent, minSlope, maxSlope)
     *
     * Recommended: Leave at defaults unless experiencing aliasing artifacts
     */
    double minSlope = -8.0;   // Minimum tangent slope
    double maxSlope = 8.0;    // Maximum tangent slope

    /**
     * Pin endpoints to (-1, -1) and (1, 1).
     *
     * When true: First and last anchors are always at domain boundaries
     * When false: Endpoints can be placed anywhere
     *
     * Recommended: true (ensures full domain coverage)
     */
    bool pinEndpoints = true;

    /**
     * Tangent computation algorithm.
     *
     * Available algorithms:
     * - FritschCarlson: Monotone-preserving with no-overshoot guarantee (default)
     * - PCHIP: Smoother but may overshoot (requires iterative correction)
     * - Akima: Local weighted average (good for noisy data)
     * - FiniteDifference: Simple baseline (for testing/comparison)
     *
     * Recommendation: FritschCarlson for production (predictable UI behavior)
     * See docs/architecture/spline-algorithm-decision.md for full analysis.
     */
    TangentAlgorithm tangentAlgorithm = TangentAlgorithm::FritschCarlson;

    /**
     * Enable anchor pruning - experimental post-processing step.
     *
     * ⚠️ DISABLED BY DEFAULT - experimental feature with known limitations.
     *
     * When enabled, iteratively removes anchors that don't significantly contribute
     * to curve accuracy. However, pruning has been found to be too aggressive:
     * - Only validates error at discrete sample points
     * - Can remove anchors preserving features between samples
     * - Breaks backtranslation stability guarantees
     *
     * Status: Adaptive tolerance (built into greedy fitting) already achieves
     * minimal anchor counts without pruning's drawbacks.
     *
     * Recommendation: Keep disabled for production use
     * See docs/architecture/spline-curve-fitting.md Section 4b for details.
     */
    bool enableAnchorPruning = false;

    /**
     * Pruning tolerance multiplier - only used if enableAnchorPruning=true.
     *
     * Multiplies the adaptive tolerance to create a more relaxed threshold for pruning.
     * Higher values = more aggressive pruning = fewer anchors = higher risk.
     *
     * Formula: pruningTolerance = adaptiveTolerance × pruningToleranceMultiplier
     *
     * Range: 1.0 (conservative) to 2.5 (aggressive)
     * Default: 1.5 (moderate, but still too aggressive in practice)
     */
    double pruningToleranceMultiplier = 1.5;

    /**
     * Enable zero-crossing drift detection (defensive DC blocking).
     *
     * After greedy fitting completes, checks if fitted spline introduced
     * DC drift at x=0 compared to original curve. Only adds corrective
     * anchor if drift exceeds tolerance.
     *
     * Philosophy: Trust the algorithm, verify the result.
     * - If base curve crosses zero at x≈0 AND fitted spline drifts → add anchor
     * - If fitted spline naturally preserves zero-crossing → no intervention
     *
     * Recommended: true (defensive protection with zero overhead when not needed)
     */
    bool enableZeroCrossingCheck = true;

    /**
     * Zero-crossing tolerance (vertical distance from base curve at x=0).
     *
     * Drift threshold: |y_fitted(0) - y_base(0)|
     * If drift > tolerance AND base curve crosses zero → add corrective anchor
     *
     * Note: Uses interpolation to handle even table sizes (no exact x=0 sample)
     *
     * Range: 0.001 (strict) to 0.05 (relaxed)
     * Default: 0.01 (1% of full range, balances protection vs intervention)
     */
    double zeroCrossingTolerance = 0.01;

    /**
     * Symmetry mode for anchor placement.
     *
     * Auto: Analyzes curve symmetry, enables paired anchors if score > threshold
     * Always: Forces symmetric anchor placement regardless of curve shape
     * Never: Disables symmetric fitting (original greedy algorithm)
     *
     * Symmetric fitting adds anchors in complementary pairs (x, -x) to
     * maintain visual symmetry for symmetric curves (tanh, x³, odd harmonics).
     *
     * Recommendation: Auto (default) - respects user intent for symmetric shapes
     */
    SymmetryMode symmetryMode = SymmetryMode::Auto;

    /**
     * Symmetry detection threshold (for Auto mode).
     *
     * If symmetry score >= threshold, enable paired anchor placement.
     *
     * Range: 0.0-1.0
     * Recommended: 0.90 (90% symmetric or better)
     */
    double symmetryThreshold = 0.90;

    // Presets
    static SplineFitConfig tight() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.002;
        cfg.derivativeTolerance = 0.05;
        cfg.maxAnchors = 128;  // Increased from 64 to allow better convergence for steep curves
        cfg.tangentAlgorithm = TangentAlgorithm::FritschCarlson;
        return cfg;
    }

    static SplineFitConfig smooth() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.01;
        cfg.derivativeTolerance = 0.02;
        cfg.maxAnchors = 24;
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
