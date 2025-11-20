#pragma once
#include "../LayeredTransferFunction.h"

namespace dsp_core {
namespace Services {

/**
 * TransferFunctionOperations - Pure service for transfer function transformations
 *
 * Provides stateless operations on LayeredTransferFunction that can be used
 * from any context (UI, presets, automation, tests).
 *
 * All operations modify the base layer and call updateComposite() to ensure
 * the visualizer reflects the changes.
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class TransferFunctionOperations {
  public:
    /**
     * Invert the base layer: f(x) → -f(x)
     *
     * Flips the transfer function vertically around the x-axis.
     * Useful for creating complementary distortion curves.
     */
    static void invert(LayeredTransferFunction& ltf);

    /**
     * Remove instantaneous DC offset
     *
     * Subtracts the value at x=0 (center of table) from all base layer values.
     * This ensures f(0) = 0 after the operation.
     *
     * Use when you want the transfer function to pass through the origin,
     * eliminating DC offset for signals that cross zero.
     */
    static void removeDCInstantaneous(LayeredTransferFunction& ltf);

    /**
     * Remove steady-state DC offset
     *
     * Subtracts the average value of the entire base layer from all values.
     * This centers the transfer function around zero.
     *
     * Use when you want to remove overall bias from the transfer function,
     * which can help reduce DC offset in the processed audio.
     */
    static void removeDCSteadyState(LayeredTransferFunction& ltf);

    /**
     * Normalize the base layer to [-1, 1] range
     *
     * Scales all base layer values so that max(|f(x)|) = 1.0.
     * This maximizes the dynamic range without clipping.
     *
     * No-op if the base layer is essentially zero (max < 1e-10).
     */
    static void normalize(LayeredTransferFunction& ltf);

  private:
    TransferFunctionOperations() = delete; // Pure static utility
};

} // namespace Services
} // namespace dsp_core
