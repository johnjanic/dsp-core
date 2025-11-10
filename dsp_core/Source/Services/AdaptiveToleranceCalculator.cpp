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
    // - At 0% capacity:   tolerance = baseline × 1.0
    // - At 50% capacity:  tolerance = baseline × 2.0 (default multiplier)
    // - At 100% capacity: tolerance = baseline × 3.0 (default multiplier)
    double adaptiveTolerance = baselineTolerance * (1.0 + anchorRatio * config.anchorDensityMultiplier);

    return adaptiveTolerance;
}

} // namespace Services
} // namespace dsp_core
