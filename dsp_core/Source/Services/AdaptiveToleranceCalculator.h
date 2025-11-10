#pragma once

namespace dsp_core {
namespace Services {

/**
 * AdaptiveToleranceCalculator - Pure service for computing adaptive error tolerances
 *
 * Calculates dynamic tolerance values that scale with:
 *   1. Vertical range (absolute error budget)
 *   2. Anchor density (prevents over-fitting as anchors accumulate)
 *
 * Purpose:
 *   Solve the "anchor creeping" problem where refitting curves produces exponentially
 *   more anchors on each cycle. Adaptive tolerance increases as anchors approach the
 *   maximum limit, relaxing fit requirements and preventing over-fitting.
 *
 * Formula:
 *   tolerance = baselineTolerance × (1 + anchorRatio × multiplier)
 *   where anchorRatio = currentAnchors / maxAnchors
 *
 * Example:
 *   - 0% anchors used:  tolerance = 1.0× baseline (tight fit)
 *   - 50% anchors used: tolerance = 2.0× baseline (moderate relaxation)
 *   - 100% anchors used: tolerance = 3.0× baseline (maximum relaxation)
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 *   - Deterministic output
 */
class AdaptiveToleranceCalculator {
public:
    /**
     * Configuration for adaptive tolerance calculation
     */
    struct Config {
        /**
         * Relative error target as fraction of vertical range (0.0-1.0)
         * Default: 0.01 (1% of vertical range)
         */
        double relativeErrorTarget;

        /**
         * Multiplier for anchor density scaling
         * Higher values increase tolerance more aggressively as anchors accumulate
         * Default: 2.0 (tolerance triples at 100% anchor capacity)
         */
        double anchorDensityMultiplier;

        /**
         * Default constructor - uses recommended defaults
         */
        Config() : relativeErrorTarget(0.01), anchorDensityMultiplier(2.0) {}
    };

    /**
     * Compute adaptive tolerance based on curve complexity and anchor density
     *
     * @param verticalRange Total vertical range of the curve (maxY - minY)
     * @param currentAnchors Number of anchors currently in use
     * @param maxAnchors Maximum allowed anchors
     * @param config Configuration parameters (optional)
     * @return Adaptive tolerance value
     */
    static double computeTolerance(double verticalRange,
                                   int currentAnchors,
                                   int maxAnchors,
                                   const Config& config = Config{});

private:
    AdaptiveToleranceCalculator() = delete;  // Pure static service
};

} // namespace Services
} // namespace dsp_core
