#include "AdaptiveToleranceCalculator.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {
namespace Services {

double AdaptiveToleranceCalculator::computeTolerance(double verticalRange,
                                                     int currentAnchors,
                                                     int maxAnchors,
                                                     const Config& config) {
    // Sanity checks
    if (maxAnchors <= 0) {
        return config.relativeErrorTarget * verticalRange;  // Fallback to baseline
    }

    if (currentAnchors < 0) {
        currentAnchors = 0;  // Clamp to zero
    }

    // Compute baseline tolerance from vertical range
    double baselineTolerance = verticalRange * config.relativeErrorTarget;

    // Compute anchor density ratio (0.0 = no anchors, 1.0 = at capacity)
    double anchorRatio = static_cast<double>(currentAnchors) / static_cast<double>(maxAnchors);
    anchorRatio = std::min(1.0, anchorRatio);  // Clamp to [0, 1]

    // Apply adaptive scaling
    // Formula: tolerance = baseline × (1 + anchorRatio × multiplier)
    // Using LINEAR scaling for better backtranslation stability at low anchor counts
    // With default multiplier=8.0:
    // - At 0% capacity:   tolerance = baseline × 1.0
    // - At 2.3% (3/128):  tolerance = baseline × 1.18   (early relaxation for backtranslation)
    // - At 10% capacity:  tolerance = baseline × 1.8
    // - At 25% capacity:  tolerance = baseline × 3.0
    // - At 50% capacity:  tolerance = baseline × 5.0
    // - At 100% capacity: tolerance = baseline × 9.0
    //
    // This is more aggressive at low anchor counts (good for backtranslation)
    // while still allowing complex curves to use their full anchor budget.
    double adaptiveTolerance = baselineTolerance * (1.0 + anchorRatio * config.anchorDensityMultiplier);

    return adaptiveTolerance;
}

} // namespace Services
} // namespace dsp_core
