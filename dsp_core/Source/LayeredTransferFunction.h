#pragma once
#include "HarmonicLayer.h"
#include "SplineLayer.h"
#include <juce_core/juce_core.h>
#include <vector>
#include <atomic>
#include <memory>

namespace dsp_core {

/**
 * RenderingMode - Determines which layer evaluation path to use
 *
 * Paint    → Direct base layer (no normalization, harmonics baked)
 * Harmonic → Base + harmonics with normalization (harmonics non-zero)
 * Spline   → Direct spline evaluation (bypasses base+harmonics)
 */
enum class RenderingMode {
    Paint,      // Direct base layer output (no normalization)
    Harmonic,   // Base + harmonics with normalization
    Spline      // Direct spline evaluation
};

/**
 * LayeredTransferFunction - Multi-layer waveshaping state container
 *
 * Stores UI state for transfer function editing: base layer (wavetable), harmonic
 * coefficients, and spline anchors. The renderer (SeamlessTransferFunction) uses
 * this state to generate production-ready LUTs at 25Hz for audio processing.
 *
 * Architecture:
 *   UI State → Renderer (25Hz) → Production LUT → Audio Thread
 *
 * Mixing formula (computed on-demand by renderer):
 *   UnNormMix[i] = wavetableCoeff × Base[i] + Σ(harmonicCoeff[n] × Harmonic_n[i])
 *   where wavetableCoeff = coefficients[0]
 *
 * Normalization (applied by renderer, not stored here):
 *   maxAbsValue = max(abs(UnNormMix[i]))
 *   normalizationScalar = 1.0 / maxAbsValue
 *   Output[i] = normalizationScalar × UnNormMix[i]
 *
 * Critical: Layers (baseTable, harmonic evaluations) are NEVER modified by normalization.
 * This allows seamless coefficient mixing in TransferFunctionController.
 *
 * Thread safety:
 *   - Base layer uses atomics for lock-free reads
 *   - All mutations should be from message thread only
 *   - Version counter tracks changes for renderer polling
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
    void clearBaseLayer(); // Set all base values to 0.0

    // Harmonic layer (for algorithm settings only)
    HarmonicLayer& getHarmonicLayer();
    const HarmonicLayer& getHarmonicLayer() const;

    // Spline layer (NEW: for spline mode)
    SplineLayer& getSplineLayer();
    const SplineLayer& getSplineLayer() const;

    // Coefficient access (includes WT mix at index 0 + harmonics at indices 1..N)
    void setCoefficient(int index, double value);
    double getCoefficient(int index) const;
    int getNumCoefficients() const {
        return static_cast<int>(coefficients.size());
    }

    /**
     * Compute composite value on-demand at specific index
     *
     * Computes: normScalar * (wtCoeff * baseValue + harmonicValue)
     *
     * This method computes the composite on-demand from UI state (base layer + harmonics).
     * Used for baking operations and legacy interpolation methods.
     *
     * Thread-safe: Can be called from any thread (reads atomics with acquire ordering)
     *
     * @param index Table index [0, tableSize)
     * @return Composite value with normalization applied, or 0.0 if index out of bounds
     */
    double computeCompositeAt(int index) const;

    // Normalization scalar (read-only access)
    double getNormalizationScalar() const {
        return normalizationScalar.load(std::memory_order_acquire);
    }

    //==========================================================================
    // Normalization Control
    //==========================================================================

    /**
     * Enable or disable deferred normalization
     *
     * When enabled, the renderer will use the frozen normalization scalar instead of
     * recalculating it. This prevents visual shifting during multi-point operations
     * like paint strokes.
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
     * Set rendering mode (determines evaluation path for LUT rendering)
     *
     * Paint    → Direct base layer (no normalization)
     * Harmonic → Base + harmonics with normalization
     * Spline   → Direct spline evaluation
     *
     * Thread-safe: Can be called from any thread
     *
     * @param mode The rendering mode to use
     */
    void setRenderingMode(RenderingMode mode);

    /**
     * Get current rendering mode
     *
     * Thread-safe: Can be called from any thread
     *
     * @return Current rendering mode
     */
    RenderingMode getRenderingMode() const;

    //==========================================================================
    // Harmonic Layer Baking
    //==========================================================================

    // Constants
    static constexpr int NUM_HARMONICS = 40;
    static constexpr int NUM_HARMONIC_COEFFICIENTS = NUM_HARMONICS + 1; // wtMix + h1..h40
    static constexpr double HARMONIC_EPSILON = 1e-6;                    // ~-120dB threshold for "effectively zero"

    /**
     * Check if any harmonic coefficients are non-zero
     *
     * Used for no-op optimization before baking.
     * Checks coefficients[1..40] (harmonics only, not WT mix at [0]).
     *
     * @return true if any harmonic amplitude exceeds HARMONIC_EPSILON
     */
    bool hasNonZeroHarmonics() const;

    /**
     * Bake harmonic layer into base layer and reset harmonics to zero
     *
     * This captures the current composite curve (base + harmonics) and writes it to base layer.
     * After baking:
     *   - Base layer contains the composite values (visually identical curve)
     *   - All harmonic coefficients are set to zero
     *   - WT mix coefficient remains at its current value
     *   - Normalization scalar is preserved (not reset)
     *
     * @return true if baking occurred (harmonics were non-zero), false if no-op
     *
     * THREAD SAFETY: Call from message thread only. The baking writes to base layer
     * and increments the version counter, triggering the renderer to generate a new
     * production LUT at the next 25Hz poll.
     */
    bool bakeHarmonicsToBase();

    /**
     * Bake composite layer (base + harmonics) into base layer and reset harmonics
     *
     * This captures the current composite curve (base + harmonics with normalization)
     * and writes it to base layer. After baking:
     *   - Base layer contains the composite values (visually identical curve)
     *   - All harmonic coefficients are set to zero
     *   - WT mix coefficient is set to 1.0 (enables base layer)
     *   - Normalization scalar will be recalculated by renderer
     *
     * THREAD SAFETY: Call from message thread only. Increments version counter
     * to trigger renderer update at next 25Hz poll.
     */
    void bakeCompositeToBase();

    /**
     * Get current harmonic coefficients for undo/redo
     *
     * @return array: [wtMix, h1, h2, ..., h40] (41 values)
     */
    std::array<double, NUM_HARMONIC_COEFFICIENTS> getHarmonicCoefficients() const;

    /**
     * Set all harmonic coefficients at once (for undo)
     *
     * @param coeffs Input array: [wtMix, h1, h2, ..., h40] (41 values)
     */
    void setHarmonicCoefficients(const std::array<double, NUM_HARMONIC_COEFFICIENTS>& coeffs);

    //==========================================================================
    // Utilities (same API as TransferFunction for compatibility)
    //==========================================================================

    int getTableSize() const {
        return tableSize;
    }
    double getMinSignalValue() const {
        return minValue;
    }
    double getMaxSignalValue() const {
        return maxValue;
    }

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
     * Evaluate base layer + harmonics explicitly (ignores spline layer)
     *
     * Used by SplineFitter to read the normalized composite when entering spline mode.
     * This ensures we fit the correct normalized curve, not the stale spline layer.
     *
     * CRITICAL: normalizationScalar is NOT reset when entering spline mode, so this
     * returns the properly normalized values that the user sees on screen.
     *
     * @param x Input value in [minValue, maxValue]
     * @return base + harmonics (with current normalization applied)
     */
    double evaluateBaseAndHarmonics(double x) const;

    /**
     * Evaluate transfer function for rendering based on current mode
     *
     * This is the primary evaluation method used by LUT renderer. It routes to the
     * appropriate evaluation path based on current rendering mode:
     *
     * Paint Mode:
     *   - Returns: wtCoeff * baseValue
     *   - No normalization applied
     *   - Harmonics assumed baked into base layer
     *
     * Harmonic Mode:
     *   - Returns: normScalar * (wtCoeff * baseValue + harmonicValue)
     *   - Normalization applied to composite
     *   - Evaluates both base and harmonic layers
     *
     * Spline Mode:
     *   - Returns: splineLayer.evaluate(x)
     *   - Direct spline evaluation
     *   - Bypasses base + harmonics entirely
     *
     * Thread-safe: Can be called from any thread (reads atomics with acquire ordering)
     *
     * @param x Normalized input in [minValue, maxValue]
     * @param normScalar Normalization scalar computed by renderer
     * @return Evaluated output value
     */
    double evaluateForRendering(double x, double normScalar) const;

    /**
     * Process block of samples in-place
     */
    void processBlock(double* samples, int numSamples) const;

    //==========================================================================
    // Interpolation/Extrapolation Modes (same as TransferFunction)
    //==========================================================================

    enum class InterpolationMode { Linear, Cubic, CatmullRom };
    enum class ExtrapolationMode { Clamp, Linear };

    void setInterpolationMode(InterpolationMode mode) {
        interpMode = mode;
        // Increment version to trigger LUT re-render with new interpolation mode
        // NOTE: Interpolation mode affects LUT evaluation, not LUT contents, but
        // we re-render anyway for simplicity (see Task 3 design rationale in
        // seamless-transfer-function-changes.md)
        versionCounter.fetch_add(1, std::memory_order_release);
    }
    void setExtrapolationMode(ExtrapolationMode mode) {
        extrapMode = mode;
        // Increment version to trigger LUT re-render with new extrapolation mode
        versionCounter.fetch_add(1, std::memory_order_release);
    }

    InterpolationMode getInterpolationMode() const {
        return interpMode;
    }
    ExtrapolationMode getExtrapolationMode() const {
        return extrapMode;
    }

    //==========================================================================
    // Version Tracking (for seamless LUT updates)
    //==========================================================================

    /**
     * Get current version of transfer function
     *
     * The version counter increments on every mutation to enable dirty detection.
     * Used by SeamlessTransferFunction to trigger asynchronous LUT rendering.
     *
     * NOTE: Version is NOT serialized - it's a runtime-only dirty tracking mechanism.
     *
     * @return Current version number
     */
    uint64_t getVersion() const {
        return versionCounter.load(std::memory_order_acquire);
    }

    /**
     * Set spline anchors (wrapper that increments version counter)
     *
     * Use this instead of getSplineLayer().setAnchors() to ensure version tracking works.
     */
    void setSplineAnchors(const std::vector<SplineAnchor>& anchors);

    /**
     * Clear all spline anchors (wrapper that increments version counter)
     */
    void clearSplineAnchors();

    /**
     * Set normalization scalar directly (for worker thread to restore frozen state)
     *
     * CRITICAL: Only used by LUT renderer to restore frozen normalization state.
     * UI code should NOT call this - use setNormalizationEnabled() instead.
     *
     * @param scalar The normalization scalar to set
     */
    void setNormalizationScalar(double scalar) {
        normalizationScalar.store(scalar, std::memory_order_release);
    }

    //==========================================================================
    // Debugging
    //==========================================================================

    /**
     * Get instance ID for debugging (identifies which LayeredTransferFunction instance this is)
     */
    int getInstanceId() const {
        return instanceId;
    }

    //==========================================================================
    // Serialization
    //==========================================================================

    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

  private:
    // Instance tracking for debugging
    static std::atomic<int> instanceCounter;
    int instanceId;

    int tableSize;
    double minValue, maxValue;

    // Layers (declare before tables since constructor initializes them first)
    std::unique_ptr<HarmonicLayer> harmonicLayer; // Harmonic basis function evaluator (no data ownership)
    std::unique_ptr<SplineLayer> splineLayer;     // NEW: Spline evaluator for spline mode

    // Coefficient storage (owned by LayeredTransferFunction)
    std::vector<double> coefficients; // [0] = WT mix, [1..N] = harmonics

    // Layers (NEVER normalized directly - preserved as-is)
    std::vector<std::atomic<double>> baseTable; // User-drawn wavetable

    // Normalization scalar (applied to mix, not to layers)
    // Mutable to allow updates from const methods (e.g., computeCompositeAt)
    mutable std::atomic<double> normalizationScalar{1.0};

    // Normalization deferral (prevents visual shifting during paint strokes)
    bool deferNormalization = false;

    // Normalization enable/disable (allows bypassing auto-normalization for creative effects)
    bool normalizationEnabled = true; // Default: enabled (safe behavior)

    // NEW: Layer mode (mutually exclusive: spline XOR harmonics)
    std::atomic<bool> splineLayerEnabled{false};

    // Rendering mode (determines evaluation path for LUT rendering)
    std::atomic<int> renderingMode{static_cast<int>(RenderingMode::Paint)};

    // Version counter for dirty detection (used by SeamlessTransferFunction)
    std::atomic<uint64_t> versionCounter{0};

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

    /**
     * Get base layer value at normalized position x using current interpolation mode
     *
     * This is a helper for evaluateForRendering() to read base layer values.
     *
     * @param x Normalized input in [minValue, maxValue]
     * @return Interpolated base layer value
     */
    double getBaseValueAt(double x) const;
};

} // namespace dsp_core
