#pragma once
#include "HarmonicLayer.h"
#include <juce_core/juce_core.h>
#include <vector>
#include <atomic>
#include <memory>
#include <functional>

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
 * Mixing formula:
 *   Composite(x) = WT_mix × Base(x) + Σ(coeff[n] × Harmonic_n(x))
 *   where WT_mix = harmonicLayer.getCoefficient(0)
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

    //==========================================================================
    // Composition
    //==========================================================================

    /**
     * Recompute composite from layers
     *
     * Call after ANY edit to base or harmonic layer.
     * This is the critical performance path - uses precomputed harmonics.
     */
    void updateComposite();

    //==========================================================================
    // Utilities (same API as TransferFunction for compatibility)
    //==========================================================================

    int getTableSize() const { return tableSize; }
    double getMinSignalValue() const { return minValue; }
    double getMaxSignalValue() const { return maxValue; }

    // Compatibility methods for TransferFunction API
    // These operate on the COMPOSITE layer (final output)
    double getTableValue(int index) const { return getCompositeValue(index); }
    void setTableValue(int index, double value);
    void applyFunction(std::function<double(double)> func);

    // Serialization compatibility
    void writeToStream(juce::OutputStream& stream) const;
    void readFromStream(juce::InputStream& stream);

    /**
     * Map table index to normalized position x ∈ [minValue, maxValue]
     */
    double normalizeIndex(int index) const;

    /**
     * Scale composite table so max(abs(value)) = 1.0
     * NOTE: Breaks layer ratios. Use soft-clipping in updateComposite() instead.
     */
    void normalizeByMaximum();

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

    // Layers
    std::vector<std::atomic<double>> baseTable;      // User-drawn
    std::unique_ptr<HarmonicLayer> harmonicLayer;    // Harmonic coefficients

    // Composite output (what audio thread reads)
    std::vector<std::atomic<double>> compositeTable;

    InterpolationMode interpMode = InterpolationMode::CatmullRom;
    ExtrapolationMode extrapMode = ExtrapolationMode::Clamp;

    // Interpolation helpers (adapted from TransferFunction)
    double interpolate(double x) const;
    double applyTransferFunctionLinear(double x) const;
    double applyTransferFunctionCubic(double x) const;
    double applyTransferFunctionCatmullRom(double x) const;
    double interpolateLinear(double y0, double y1, double t) const;
    double interpolateCubic(double y0, double y1, double y2, double y3, double t) const;
    double interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) const;
    double getSample(int i) const;
};

} // namespace dsp_core
