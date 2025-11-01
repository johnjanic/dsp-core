#pragma once
#include "HarmonicLayer.h"
#include "SplineLayer.h"
#include <juce_core/juce_core.h>
#include <vector>
#include <atomic>
#include <memory>

namespace dsp_core {

/**
 * LayeredTransferFunction - Multi-layer waveshaping compositor
 *
 * Combines a user-drawn base layer with a harmonic layer to produce
 * a composite transfer function for audio processing.
 *
 * Architecture:
 *   Base Layer (wavetable) + Harmonic Layer (coefficients) → Composite Table
 *
 * Mixing formula (unnormalized):
 *   UnNormMix[i] = wavetableCoeff × Base[i] + Σ(harmonicCoeff[n] × Harmonic_n[i])
 *   where wavetableCoeff = harmonicLayer.getCoefficient(0)
 *
 * Normalization (applied separately to preserve layer data):
 *   maxAbsValue = max(abs(UnNormMix[i]))
 *   normalizationScalar = 1.0 / maxAbsValue
 *   Composite[i] = normalizationScalar × UnNormMix[i]
 *
 * Critical: Layers (baseTable, harmonic evaluations) are NEVER modified by normalization.
 * This allows seamless coefficient mixing in TransferFunctionController.
 *
 * Thread safety:
 *   - Base layer uses atomics for lock-free reads/writes
 *   - Composite uses atomics for lock-free reads by audio thread
 *   - updateComposite() should be called from UI thread only
 */
class LayeredTransferFunction {
public:
    LayeredTransferFunction(int tableSize, double minVal, double maxVal);

    //==========================================================================
    // Layer Access
    //==========================================================================

    // Base layer (user-drawn wavetable)
    double getBaseLayerValue(int index) const;
    void setBaseLayerValue(int index, double value);
    void clearBaseLayer();  // Set all base values to 0.0

    // Harmonic layer (for algorithm settings only)
    HarmonicLayer& getHarmonicLayer();
    const HarmonicLayer& getHarmonicLayer() const;

    // Spline layer (NEW: for spline mode)
    SplineLayer& getSplineLayer();
    const SplineLayer& getSplineLayer() const;

    // Coefficient access (includes WT mix at index 0 + harmonics at indices 1..N)
    void setCoefficient(int index, double value);
    double getCoefficient(int index) const;
    int getNumCoefficients() const { return static_cast<int>(coefficients.size()); }

    // Composite (final output for audio processing)
    double getCompositeValue(int index) const;

    // Normalization scalar (read-only access)
    double getNormalizationScalar() const { return normalizationScalar.load(std::memory_order_acquire); }

    //==========================================================================
    // Composition
    //==========================================================================

    /**
     * Recompute composite from layers
     *
     * Call after ANY edit to base or harmonic layer coefficients.
     *
     * Algorithm:
     *   1. Compute unnormalized mix: UnNorm[i] = wtCoeff*Base[i] + Σ(hCoeff[n]*H_n[i])
     *   2. Find maxAbsValue = max(abs(UnNorm[i]))
     *   3. Compute normalizationScalar = 1.0 / maxAbsValue
     *   4. Store normalized composite: Composite[i] = normalizationScalar * UnNorm[i]
     *
     * Critical: Base layer and harmonic evaluations remain unchanged.
     *
     * If normalization is deferred (via setDeferNormalization), the normalization scalar
     * remains frozen and the composite is computed using the existing scalar. This prevents
     * visual shifting during paint strokes.
     */
    void updateComposite();

    /**
     * Enable or disable deferred normalization
     *
     * When enabled, updateComposite() will freeze the normalization scalar and only
     * update the composite table using the existing scalar. This prevents visual shifting
     * during multi-point operations like paint strokes.
     *
     * When disabled, normalization resumes normal behavior (recalculating the scalar).
     *
     * @param shouldDefer If true, freeze normalization scalar; if false, resume normal behavior
     */
    void setDeferNormalization(bool shouldDefer);

    /**
     * Check if normalization is currently deferred
     */
    bool isNormalizationDeferred() const;

    /**
     * Enable or disable automatic normalization
     *
     * When disabled, the normalization scalar is fixed at 1.0, effectively bypassing
     * the automatic scaling that keeps output in [-1, 1]. This allows for creative
     * distortion effects or preserving exact mathematical relationships.
     *
     * WARNING: Disabling normalization can result in output values > ±1.0, which may
     * cause clipping or distortion in the audio output.
     *
     * @param enabled If true, enable automatic normalization (default); if false, bypass normalization
     */
    void setNormalizationEnabled(bool enabled);

    /**
     * Check if automatic normalization is enabled
     */
    bool isNormalizationEnabled() const;

    /**
     * Enable or disable spline layer (mutually exclusive with harmonic layer)
     *
     * When enabled:
     *   - Audio thread uses direct spline evaluation
     *   - Normalization is locked to 1.0 (identity)
     *   - Cache is invalidated
     *
     * When disabled:
     *   - Audio thread uses base + harmonics
     *   - Normalization resumes normal behavior
     *
     * CRITICAL: Call bakeCompositeToBase() before enabling spline layer
     * to ensure harmonics are zeroed.
     *
     * @param enabled If true, enable spline mode; if false, use harmonic mode
     */
    void setSplineLayerEnabled(bool enabled);

    /**
     * Check if spline layer is currently enabled
     */
    bool isSplineLayerEnabled() const;

    /**
     * Invalidate composite cache (forces direct evaluation)
     *
     * Called when:
     *   - Spline anchors change
     *   - Base layer edited
     *   - Coefficients changed
     *   - Layer mode switched
     */
    void invalidateCompositeCache();

    /**
     * Check if composite cache is valid
     *
     * @return true if cached path can be used, false if direct evaluation required
     */
    bool isCompositeCacheValid() const;

    //==========================================================================
    // Utilities (same API as TransferFunction for compatibility)
    //==========================================================================

    int getTableSize() const { return tableSize; }
    double getMinSignalValue() const { return minValue; }
    double getMaxSignalValue() const { return maxValue; }

    /**
     * Map table index to normalized position x ∈ [minValue, maxValue]
     */
    double normalizeIndex(int index) const;

    //==========================================================================
    // Processing (reads composite - thread-safe)
    //==========================================================================

    /**
     * Apply transfer function to input sample
     * Uses composite table with interpolation
     */
    double applyTransferFunction(double x) const;

    /**
     * Process block of samples in-place
     */
    void processBlock(double* samples, int numSamples);

    //==========================================================================
    // Interpolation/Extrapolation Modes (same as TransferFunction)
    //==========================================================================

    enum class InterpolationMode { Linear, Cubic, CatmullRom };
    enum class ExtrapolationMode { Clamp, Linear };

    void setInterpolationMode(InterpolationMode mode) { interpMode = mode; }
    void setExtrapolationMode(ExtrapolationMode mode) { extrapMode = mode; }

    InterpolationMode getInterpolationMode() const { return interpMode; }
    ExtrapolationMode getExtrapolationMode() const { return extrapMode; }

    //==========================================================================
    // Serialization
    //==========================================================================

    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

private:
    int tableSize;
    double minValue, maxValue;

    // Layers (declare before tables since constructor initializes them first)
    std::unique_ptr<HarmonicLayer> harmonicLayer;    // Harmonic basis function evaluator (no data ownership)
    std::unique_ptr<SplineLayer> splineLayer;        // NEW: Spline evaluator for spline mode

    // Coefficient storage (owned by LayeredTransferFunction)
    std::vector<double> coefficients;  // [0] = WT mix, [1..N] = harmonics

    // Layers (NEVER normalized directly - preserved as-is)
    std::vector<std::atomic<double>> baseTable;      // User-drawn wavetable

    // Composite output (what audio thread reads)
    std::vector<std::atomic<double>> compositeTable;

    // Normalization scalar (applied to mix, not to layers)
    std::atomic<double> normalizationScalar{ 1.0 };

    // Normalization deferral (prevents visual shifting during paint strokes)
    bool deferNormalization = false;

    // Normalization enable/disable (allows bypassing auto-normalization for creative effects)
    bool normalizationEnabled = true;  // Default: enabled (safe behavior)

    // NEW: Layer mode (mutually exclusive: spline XOR harmonics)
    std::atomic<bool> splineLayerEnabled{false};

    // NEW: Cache validity flag (allows direct evaluation when cache invalid)
    std::atomic<bool> compositeCacheValid{false};

    // Pre-allocated scratch buffer for updateComposite() (eliminates heap allocation)
    mutable std::vector<double> unnormalizedMixBuffer;

    InterpolationMode interpMode = InterpolationMode::CatmullRom;
    ExtrapolationMode extrapMode = ExtrapolationMode::Clamp;

    // Interpolation helpers with dual-path optimization
    // Each method has two code paths selected by a single branch on extrapMode:
    //   - Fast path (Clamp): Direct clamped loads, relaxed memory order (default)
    //   - Slow path (Linear): Boundary checks + slope calculations for extrapolation
    double interpolate(double x) const;
    double interpolateLinear(double x) const;
    double interpolateCubic(double x) const;
    double interpolateCatmullRom(double x) const;

    // NEW: Direct evaluation (bypasses composite cache)
    // Used when cache is invalid (during anchor drag, after edits)
    double evaluateDirect(double x) const;

    // NEW: Interpolate base layer directly (bypasses composite)
    // Needed for harmonic mode when cache invalid
    double interpolateBase(double x) const;

    // Mode-specific composite update helpers
    void updateCompositeSplineMode();
    void updateCompositeHarmonicMode();

    // Normalization computation (extracted for clarity)
    double computeNormalizationScalar(double maxAbsValue);

    // Validation helper
    bool hasNonZeroHarmonics() const;
};

} // namespace dsp_core
