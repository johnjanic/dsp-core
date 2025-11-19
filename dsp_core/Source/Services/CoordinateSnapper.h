#pragma once
#include <juce_core/juce_core.h>

namespace dsp_core {
namespace Services {

/**
 * CoordinateSnapper - Pure service for grid snapping in world space
 *
 * Provides pure static methods for snapping coordinates to grid multiples.
 * All snapping operations occur in world space (zoom-independent) with
 * configurable grid step sizes.
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 *   - Complex math logic
 *   - No side effects
 */
class CoordinateSnapper {
  public:
    /**
     * Snap single value to nearest grid multiple
     *
     * @param value The value to snap (in world space)
     * @param gridStep The grid spacing (in world units)
     * @return Snapped value: round(value / gridStep) * gridStep
     */
    static double snapValue(double value, double gridStep);

    /**
     * Snap 2D point with per-axis control
     *
     * @param point The point to snap (in world space)
     * @param gridStep The grid spacing (in world units)
     * @param snapX Enable snapping on X axis
     * @param snapY Enable snapping on Y axis
     * @return Snapped point with selected axes snapped to grid
     */
    static juce::Point<double> snapPoint(const juce::Point<double>& point, double gridStep, bool snapX = true,
                                         bool snapY = true);

    /**
     * Check if value is within threshold of a grid line
     *
     * @param value The value to check (in world space)
     * @param gridStep The grid spacing (in world units)
     * @param thresholdWorldSpace Maximum distance to grid line (in world units)
     * @return true if value is within threshold of nearest grid line
     */
    static bool isNearGridLine(double value, double gridStep, double thresholdWorldSpace);

    /**
     * Get nearest grid line value
     *
     * @param value The value to check (in world space)
     * @param gridStep The grid spacing (in world units)
     * @return The nearest grid line value
     */
    static double nearestGridLine(double value, double gridStep);

  private:
    CoordinateSnapper() = delete; // Pure static utility
};

} // namespace Services
} // namespace dsp_core
