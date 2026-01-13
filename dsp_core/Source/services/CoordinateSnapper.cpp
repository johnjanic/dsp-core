#include "CoordinateSnapper.h"
#include <cmath>

namespace dsp_core::Services {

double CoordinateSnapper::snapValue(double value, double gridStep) {
    // Return original value for invalid grid steps
    if (gridStep <= 0.0) {
        return value;
    }

    // Snap to nearest grid multiple using stable rounding
    // Formula: round(value / gridStep) * gridStep
    return std::round(value / gridStep) * gridStep;
}

juce::Point<double> CoordinateSnapper::snapPoint(const juce::Point<double>& point, double gridStep, bool snapX,
                                                 bool snapY) {
    return {snapX ? snapValue(point.x, gridStep) : point.x, snapY ? snapValue(point.y, gridStep) : point.y};
}

bool CoordinateSnapper::isNearGridLine(double value, double gridStep, double thresholdWorldSpace) {
    // Return false for invalid grid steps
    if (gridStep <= 0.0) {
        return false;
    }

    // Find nearest grid line and check distance
    const double nearest = nearestGridLine(value, gridStep);
    return std::abs(value - nearest) <= thresholdWorldSpace;
}

double CoordinateSnapper::nearestGridLine(double value, double gridStep) {
    // Return original value for invalid grid steps
    if (gridStep <= 0.0) {
        return value;
    }

    // Same logic as snapValue - round to nearest grid multiple
    return std::round(value / gridStep) * gridStep;
}

} // namespace dsp_core::Services
