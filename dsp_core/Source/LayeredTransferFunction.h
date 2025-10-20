#pragma once
#include "HarmonicLayer.h"
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

    // Harmonic layer
    HarmonicLayer& getHarmonicLayer();
    const HarmonicLayer& getHarmonicLayer() const;

    // Composite (final output for audio processing)
    double getCompositeValue(int index) const;

    // Normalization scalar (read-only access)
    double getNormalizationScalar() const { return normalizationScalar.load(); }

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

    // Harmonic layer (declare before tables since constructor initializes it first)
    std::unique_ptr<HarmonicLayer> harmonicLayer;    // Harmonic coefficients + basis functions

    // Layers (NEVER normalized directly - preserved as-is)
    std::vector<std::atomic<double>> baseTable;      // User-drawn wavetable

    // Composite output (what audio thread reads)
    std::vector<std::atomic<double>> compositeTable;

    // Normalization scalar (applied to mix, not to layers)
    std::atomic<double> normalizationScalar{ 1.0 };

    // Normalization deferral (prevents visual shifting during paint strokes)
    bool deferNormalization = false;

    InterpolationMode interpMode = InterpolationMode::CatmullRom;
    ExtrapolationMode extrapMode = ExtrapolationMode::Clamp;

    // Interpolation helpers (copy from TransferFunction implementation)
    double interpolate(double x) const;
    double interpolateLinear(double x) const;
    double interpolateCubic(double x) const;
    double interpolateCatmullRom(double x) const;
};

} // namespace dsp_core
