#pragma once
#include "SplineTypes.h"
#include <plugin-core/PropertyTree.h>
#include <vector>
#include <atomic>
#include <memory>

namespace dsp_core {

/**
 * SplineLayer - PCHIP spline evaluator for LayeredTransferFunction
 *
 * Provides real-time spline evaluation without wavetable quantization.
 * Uses atomic shared_ptr for lock-free anchor updates from UI thread.
 *
 * Thread Safety:
 *   - UI thread: setAnchors() swaps atomic shared_ptr
 *   - Audio thread: evaluate() reads current shared_ptr (lock-free)
 *
 * Pattern: Mirrors HarmonicLayer (pure evaluator, no table storage)
 */
class SplineLayer {
  public:
    SplineLayer();

    // Anchor Management (UI thread only)

    /**
     * Set spline anchors (atomic swap for lock-free reads)
     *
     * Thread-safe: Can be called from UI thread while audio thread reads.
     *
     * @param anchors Control points with computed tangents (call SplineFitter::computeTangents first)
     */
    void setAnchors(const std::vector<SplineAnchor>& anchors);

    /**
     * Get current anchors (atomic load)
     *
     * @return Copy of current anchor vector (safe snapshot)
     */
    std::vector<SplineAnchor> getAnchors() const;

    // Evaluation (thread-safe, lock-free reads)

    /**
     * Evaluate spline at x using PCHIP cubic Hermite interpolation
     *
     * Thread-safe: Can be called from audio thread while UI thread updates anchors.
     *
     * @param x Input value (typically in [-1, 1] but extrapolation supported)
     * @return Interpolated y value
     */
    double evaluate(double x) const;

    // Serialization
    plugin::PropertyTree toPropertyTree() const;
    void fromPropertyTree(const plugin::PropertyTree& tree);

  private:
    // Lock-free anchor storage (C++17 compatible)
    // Uses std::atomic_load/store for thread-safe shared_ptr operations
    // Audio thread reads via atomic_load, UI thread writes via atomic_store
    std::shared_ptr<const std::vector<SplineAnchor>> anchorsPtr;
};

} // namespace dsp_core
