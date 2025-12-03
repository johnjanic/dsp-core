#include "LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>

// Branch prediction hints (improve CPU branch predictor performance)
// GCC/Clang support __builtin_expect for optimization
#if defined(__GNUC__) || defined(__clang__)
#define juce_likely(x) __builtin_expect(!!(x), 1)
#define juce_unlikely(x) __builtin_expect(!!(x), 0)
#else
#define juce_likely(x) (x)
#define juce_unlikely(x) (x)
#endif

namespace dsp_core {

// Static instance counter for debugging
std::atomic<int> LayeredTransferFunction::instanceCounter{0};

namespace {
    // Number of harmonics for harmonic layer synthesis
    constexpr int kNumHarmonics = 40;
    // Total coefficients: [0] = wavetable mix, [1..40] = harmonic amplitudes
    constexpr int kTotalCoefficients = kNumHarmonics + 1;
    // Epsilon for zero comparison in normalization
    constexpr double kNormalizationEpsilon = 1e-12;
} // namespace

LayeredTransferFunction::LayeredTransferFunction(int tableSize, double minVal, double maxVal)
    : instanceId(instanceCounter.fetch_add(1, std::memory_order_relaxed)),
      tableSize(tableSize), minValue(minVal), maxValue(maxVal), harmonicLayer(std::make_unique<HarmonicLayer>(kNumHarmonics)),
      splineLayer(std::make_unique<SplineLayer>()), // NEW: Initialize spline layer
      coefficients(kTotalCoefficients, 0.0),        // kTotalCoefficients: [0] = WT, [1..40] = harmonics
      baseTable(tableSize) {

    DBG("[MODEL:" + juce::String(instanceId) + "] LayeredTransferFunction instance created");

    // Initialize coefficients
    coefficients[0] = 1.0; // Default WT mix = 1.0 (full base layer)
    // coefficients[1..40] already initialized to 0.0

    // Initialize base layer to identity: y = x
    for (int i = 0; i < tableSize; ++i) {
        const double x = normalizeIndex(i);
        baseTable[i].store(x, std::memory_order_release);
    }

    // Precompute harmonic basis functions
    harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);

    // No need to compute composite here - renderer will generate it at 25Hz
}

double LayeredTransferFunction::getBaseLayerValue(int index) const {
    if (index >= 0 && index < tableSize) {
        const double value = baseTable[index].load(std::memory_order_acquire);
        return value;
    }
    return 0.0;
}

void LayeredTransferFunction::setBaseLayerValue(int index, double value) {
    if (index >= 0 && index < tableSize) {
        baseTable[index].store(value, std::memory_order_release);

        // Debug: Log a sample of base layer writes at center point
        if (index == 8192) {
            DBG("[MODEL:" + juce::String(instanceId) + "] setBaseLayerValue(" + juce::String(index) + ", " + juce::String(value) + ")");
        }

        versionCounter.fetch_add(1, std::memory_order_release);
    }
}

void LayeredTransferFunction::clearBaseLayer() {
    for (int i = 0; i < tableSize; ++i) {
        baseTable[i].store(0.0, std::memory_order_release);
    }
    versionCounter.fetch_add(1, std::memory_order_release);
}

HarmonicLayer& LayeredTransferFunction::getHarmonicLayer() {
    return *harmonicLayer;
}

const HarmonicLayer& LayeredTransferFunction::getHarmonicLayer() const {
    return *harmonicLayer;
}

SplineLayer& LayeredTransferFunction::getSplineLayer() {
    return *splineLayer;
}

const SplineLayer& LayeredTransferFunction::getSplineLayer() const {
    return *splineLayer;
}

void LayeredTransferFunction::setSplineAnchors(const std::vector<SplineAnchor>& anchors) {
    splineLayer->setAnchors(anchors);
    versionCounter.fetch_add(1, std::memory_order_release);
}

void LayeredTransferFunction::clearSplineAnchors() {
    splineLayer->setAnchors({});
    versionCounter.fetch_add(1, std::memory_order_release);
}

void LayeredTransferFunction::setCoefficient(int index, double value) {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        coefficients[index] = value;

        versionCounter.fetch_add(1, std::memory_order_release);
    }
}

double LayeredTransferFunction::getCoefficient(int index) const {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        return coefficients[index];
    }
    return 0.0;
}

double LayeredTransferFunction::computeCompositeAt(int index) const {
    // Bounds check
    if (index < 0 || index >= tableSize) {
        return 0.0;
    }

    // Normalize index to x-coordinate
    const double x = normalizeIndex(index);

    // Read base layer value
    const double baseValue = baseTable[index].load(std::memory_order_acquire);

    // Evaluate harmonics at x
    const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);

    // Compute unnormalized composite: wtCoeff * base + harmonics
    const double wtCoeff = coefficients[0];
    const double unnormalized = wtCoeff * baseValue + harmonicValue;

    // Apply normalization if enabled
    if (!normalizationEnabled) {
        return unnormalized;
    }

    // If normalization is deferred, use frozen scalar
    if (deferNormalization) {
        const double normScalar = normalizationScalar.load(std::memory_order_acquire);
        return normScalar * unnormalized;
    }

    // Otherwise, compute normalization scalar on-the-fly
    // This requires scanning the entire table to find max absolute value
    double maxAbsValue = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        const double xi = normalizeIndex(i);
        const double baseVal = baseTable[i].load(std::memory_order_acquire);
        const double harmVal = harmonicLayer->evaluate(xi, coefficients, tableSize);
        const double unnorm = wtCoeff * baseVal + harmVal;
        maxAbsValue = std::max(maxAbsValue, std::abs(unnorm));
    }

    constexpr double kNormalizationEpsilon = 1e-12;
    if (maxAbsValue < kNormalizationEpsilon) {
        return unnormalized;
    }

    const double normScalar = 1.0 / maxAbsValue;

    // Store the computed scalar so getNormalizationScalar() returns a meaningful value
    // This is safe to do from a const method because normalizationScalar is atomic
    normalizationScalar.store(normScalar, std::memory_order_release);

    return normScalar * unnormalized;
}

void LayeredTransferFunction::setDeferNormalization(bool shouldDefer) {
    deferNormalization = shouldDefer;

    // When exiting deferred mode, recompute and store normalization scalar
    if (!shouldDefer && normalizationEnabled) {
        // Compute max absolute value across entire composite
        double maxAbsValue = 0.0;
        const double wtCoeff = coefficients[0];

        for (int i = 0; i < tableSize; ++i) {
            const double x = normalizeIndex(i);
            const double baseValue = baseTable[i].load(std::memory_order_acquire);
            const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
            const double unnormalized = wtCoeff * baseValue + harmonicValue;
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }

        // Update normalization scalar
        constexpr double kNormalizationEpsilon = 1e-12;
        const double normScalar = (maxAbsValue > kNormalizationEpsilon) ? (1.0 / maxAbsValue) : 1.0;
        normalizationScalar.store(normScalar, std::memory_order_release);

        const uint64_t newVersion = versionCounter.fetch_add(1, std::memory_order_release) + 1;
        DBG("[MODEL:" + juce::String(instanceId) + "] Paint stroke complete, version incremented to " + juce::String(static_cast<int64_t>(newVersion)));
    }
}

bool LayeredTransferFunction::isNormalizationDeferred() const {
    return deferNormalization;
}

void LayeredTransferFunction::setNormalizationEnabled(bool enabled) {
    normalizationEnabled = enabled;

    // Increment version to trigger renderer update with new normalization state
    versionCounter.fetch_add(1, std::memory_order_release);
}

bool LayeredTransferFunction::isNormalizationEnabled() const {
    return normalizationEnabled;
}

void LayeredTransferFunction::setSplineLayerEnabled(bool enabled) {
    // Legacy API - delegates to RenderingMode for consistency
    // When enabled: use Spline rendering mode
    // When disabled: use Paint rendering mode (harmonics should be baked)
    //
    // Entering spline mode:
    // - Harmonics are kept intact (hidden but present for undo)
    // - Spline will fit to the current normalized composite (base + harmonics)
    // - Spline evaluation bypasses normalization (user controls amplitude via anchors)
    //
    // CRITICAL: We do NOT modify normalizationScalar here!
    // - normalizationScalar only affects harmonic mode evaluation
    // - Spline mode evaluation doesn't use normalizationScalar (direct PCHIP)
    // - SplineFitter needs the correct normScalar to read normalized composite
    // - Premature reset to 1.0 caused bug where SplineFitter fit unnormalized values

    // Delegate to RenderingMode (source of truth)
    setRenderingMode(enabled ? RenderingMode::Spline : RenderingMode::Paint);

    // Also update legacy flag for backward compatibility in RenderJob serialization
    splineLayerEnabled.store(enabled, std::memory_order_release);
}

bool LayeredTransferFunction::hasNonZeroHarmonics() const {
    // Check harmonics only (coefficients[1..40]), not WT mix at [0]
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        if (std::abs(coefficients[i]) > HARMONIC_EPSILON) {
            return true;
        }
    }
    return false;
}

bool LayeredTransferFunction::bakeHarmonicsToBase() {
    // Early exit if no harmonics to bake (no-op optimization)
    if (!hasNonZeroHarmonics()) {
        return false;
    }

    // CRITICAL: Compute and freeze normalization scalar BEFORE modifying coefficients
    // This ensures baked values match what the user sees on screen
    if (normalizationEnabled) {
        double maxAbsValue = 0.0;
        const double wtCoeff = coefficients[0];

        for (int i = 0; i < tableSize; ++i) {
            const double x = normalizeIndex(i);
            const double baseValue = baseTable[i].load(std::memory_order_acquire);
            const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
            const double unnormalized = wtCoeff * baseValue + harmonicValue;
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }

        constexpr double kNormalizationEpsilon = 1e-12;
        const double normScalar = (maxAbsValue > kNormalizationEpsilon) ? (1.0 / maxAbsValue) : 1.0;
        normalizationScalar.store(normScalar, std::memory_order_release);

        // Enable deferred normalization so computeCompositeAt() uses the frozen scalar
        deferNormalization = true;
    }

    // Capture composite curve (base + harmonics with normalization applied)
    // This preserves the visual appearance of the curve exactly
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = computeCompositeAt(i);  // Uses frozen normalization scalar
        setBaseLayerValue(i, compositeValue);
    }

    // Set WT coefficient to 1.0 (CRITICAL: enables the baked base layer)
    // If WT was 0 during harmonic editing, the baked base layer would be invisible
    coefficients[0] = 1.0;

    // Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Disable deferred normalization to allow recalculation
    // The baked base layer contains normalized values, so the new normalization scalar
    // will be ~1.0, and visual continuity is preserved because:
    //   composite = normScalar_new * (WT * baked_base)
    //            = normScalar_new * (1.0 * (normScalar_old * unnormalized))
    //            ≈ 1.0 * (normScalar_old * unnormalized)  [since normScalar_new ≈ 1.0]
    if (normalizationEnabled) {
        setDeferNormalization(false);  // Recalculates and stores new normalization scalar
    }

    // Increment version to trigger renderer update
    // Renderer will generate composite from base layer with WT mix = 1.0
    versionCounter.fetch_add(1, std::memory_order_release);

    return true;
}

void LayeredTransferFunction::bakeCompositeToBase() {
    // CRITICAL: Compute and freeze normalization scalar BEFORE modifying coefficients
    // This ensures baked values match what the user sees on screen
    if (normalizationEnabled) {
        double maxAbsValue = 0.0;
        const double wtCoeff = coefficients[0];

        for (int i = 0; i < tableSize; ++i) {
            const double x = normalizeIndex(i);
            const double baseValue = baseTable[i].load(std::memory_order_acquire);
            const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
            const double unnormalized = wtCoeff * baseValue + harmonicValue;
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }

        constexpr double kNormalizationEpsilon = 1e-12;
        const double normScalar = (maxAbsValue > kNormalizationEpsilon) ? (1.0 / maxAbsValue) : 1.0;
        normalizationScalar.store(normScalar, std::memory_order_release);

        // Enable deferred normalization so computeCompositeAt() uses the frozen scalar
        deferNormalization = true;
    }

    // Capture composite curve (base + harmonics with normalization applied)
    // This preserves the visual appearance of the curve exactly
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = computeCompositeAt(i);  // Uses frozen normalization scalar
        setBaseLayerValue(i, compositeValue);
    }

    // Set WT coefficient to 1.0 (enable base layer)
    coefficients[0] = 1.0;

    // Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Disable deferred normalization to allow recalculation
    // The baked base layer contains normalized values, so the new normalization scalar
    // will be ~1.0, and visual continuity is preserved
    if (normalizationEnabled) {
        setDeferNormalization(false);  // Recalculates and stores new normalization scalar
    }

    // Increment version to trigger renderer update
    // Renderer will generate composite from base layer with WT mix = 1.0
    versionCounter.fetch_add(1, std::memory_order_release);
}

std::array<double, LayeredTransferFunction::NUM_HARMONIC_COEFFICIENTS>
LayeredTransferFunction::getHarmonicCoefficients() const {
    std::array<double, NUM_HARMONIC_COEFFICIENTS> coeffs{};

    // Copy all coefficients: [0] = WT mix, [1..40] = harmonics
    for (int i = 0; i < NUM_HARMONIC_COEFFICIENTS && i < static_cast<int>(coefficients.size()); ++i) {
        coeffs[i] = coefficients[i];
    }

    return coeffs;
}

void LayeredTransferFunction::setHarmonicCoefficients(const std::array<double, NUM_HARMONIC_COEFFICIENTS>& coeffs) {
    // Set all coefficients: [0] = WT mix, [1..40] = harmonics
    for (int i = 0; i < NUM_HARMONIC_COEFFICIENTS && i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = coeffs[i];
    }

    // Increment version to trigger renderer update
    versionCounter.fetch_add(1, std::memory_order_release);
}

bool LayeredTransferFunction::isSplineLayerEnabled() const {
    // Legacy API - delegates to RenderingMode (source of truth)
    return getRenderingMode() == RenderingMode::Spline;
}

void LayeredTransferFunction::setRenderingMode(RenderingMode mode) {
    const auto oldMode = getRenderingMode();
    renderingMode.store(static_cast<int>(mode), std::memory_order_release);
    versionCounter.fetch_add(1, std::memory_order_release);

    // DIAGNOSTIC: Log rendering mode changes to track bug
    const char* oldModeStr = (oldMode == RenderingMode::Paint) ? "Paint" :
                             (oldMode == RenderingMode::Harmonic) ? "Harmonic" : "Spline";
    const char* newModeStr = (mode == RenderingMode::Paint) ? "Paint" :
                             (mode == RenderingMode::Harmonic) ? "Harmonic" : "Spline";
    DBG("[LTF:" + juce::String(instanceId) + "] setRenderingMode: " +
        juce::String(oldModeStr) + " → " + juce::String(newModeStr) +
        " (version=" + juce::String(static_cast<int64_t>(versionCounter.load())) + ")");
}

RenderingMode LayeredTransferFunction::getRenderingMode() const {
    return static_cast<RenderingMode>(renderingMode.load(std::memory_order_acquire));
}

double LayeredTransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize) {
        return 0.0;
    }
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minValue, maxValue);
}

double LayeredTransferFunction::applyTransferFunction(double x) const {
    return interpolate(x);
}

void LayeredTransferFunction::processBlock(double* samples, int numSamples) const {
    for (int i = 0; i < numSamples; ++i) {
        samples[i] = applyTransferFunction(samples[i]);
    }
}

double LayeredTransferFunction::interpolate(double x) const {
    switch (interpMode) {
    case InterpolationMode::Linear:
        return interpolateLinear(x);
    case InterpolationMode::Cubic:
        return interpolateCubic(x);
    case InterpolationMode::CatmullRom:
        return interpolateCatmullRom(x);
    default:
        return interpolateLinear(x);
    }
}

double LayeredTransferFunction::interpolateLinear(double x) const {
    // Map x from signal range to table index
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    // Computes on-demand from base layer + harmonics (no cached composite table)
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index);
        const int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

        const double y0 = computeCompositeAt(idx0);
        const double y1 = computeCompositeAt(idx1);

        return y0 + t * (y1 - y0);
    }

    // Linear extrapolation path (requires boundary checks and slope calculations)
    double y0;
    double y1;

    // Handle index
    if (index < 0) {
        const double slope = computeCompositeAt(1) - computeCompositeAt(0);
        y0 = computeCompositeAt(0) + slope * index;
    } else if (index >= tableSize) {
        const double slope = computeCompositeAt(tableSize - 1) - computeCompositeAt(tableSize - 2);
        y0 = computeCompositeAt(tableSize - 1) + slope * (index - tableSize + 1);
    } else {
        y0 = computeCompositeAt(index);
    }

    // Handle index + 1
    const int index1 = index + 1;
    if (index1 < 0) {
        const double slope = computeCompositeAt(1) - computeCompositeAt(0);
        y1 = computeCompositeAt(0) + slope * index1;
    } else if (index1 >= tableSize) {
        const double slope = computeCompositeAt(tableSize - 1) - computeCompositeAt(tableSize - 2);
        y1 = computeCompositeAt(tableSize - 1) + slope * (index1 - tableSize + 1);
    } else {
        y1 = computeCompositeAt(index1);
    }

    return y0 + t * (y1 - y0);
}

double LayeredTransferFunction::interpolateCubic(double x) const {
    // Map x from signal range to table index
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    // Computes on-demand from base layer + harmonics (no cached composite table)
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        const int idx1 = juce::jlimit(0, tableSize - 1, index);
        const int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        const int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        const double y0 = computeCompositeAt(idx0);
        const double y1 = computeCompositeAt(idx1);
        const double y2 = computeCompositeAt(idx2);
        const double y3 = computeCompositeAt(idx3);

        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            const double slope = computeCompositeAt(1) - computeCompositeAt(0);
            return computeCompositeAt(0) + slope * i;
        }
        if (i >= tableSize) {
            const double slope = computeCompositeAt(tableSize - 1) - computeCompositeAt(tableSize - 2);
            return computeCompositeAt(tableSize - 1) + slope * (i - tableSize + 1);
        }
        return computeCompositeAt(i);
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    const double a0 = y3 - y2 - y0 + y1;
    const double a1 = y0 - y1 - a0;
    const double a2 = y2 - y0;
    const double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double LayeredTransferFunction::interpolateCatmullRom(double x) const {
    // Map x from signal range to table index
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    // Computes on-demand from base layer + harmonics (no cached composite table)
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        const int idx1 = juce::jlimit(0, tableSize - 1, index);
        const int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        const int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        const double y0 = computeCompositeAt(idx0);
        const double y1 = computeCompositeAt(idx1);
        const double y2 = computeCompositeAt(idx2);
        const double y3 = computeCompositeAt(idx3);

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            const double slope = computeCompositeAt(1) - computeCompositeAt(0);
            return computeCompositeAt(0) + slope * i;
        }
        if (i >= tableSize) {
            const double slope = computeCompositeAt(tableSize - 1) - computeCompositeAt(tableSize - 2);
            return computeCompositeAt(tableSize - 1) + slope * (i - tableSize + 1);
        }
        return computeCompositeAt(i);
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    // Standard Catmull-Rom interpolation formula
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double LayeredTransferFunction::getBaseValueAt(double x) const {
    // Map x to table index
    const double tableIndex = ((x - minValue) / (maxValue - minValue)) * (tableSize - 1);

    // Handle out-of-bounds based on extrapolation mode
    if (extrapMode == ExtrapolationMode::Clamp) {
        if (tableIndex <= 0.0)
            return baseTable[0].load(std::memory_order_relaxed);
        if (tableIndex >= tableSize - 1)
            return baseTable[tableSize - 1].load(std::memory_order_relaxed);
    }

    // Get integer and fractional parts
    const int idx = static_cast<int>(tableIndex);
    const double frac = tableIndex - idx;

    // Bounds check
    if (idx < 0 || idx >= tableSize - 1) {
        // Linear extrapolation (only reached if extrapMode == Linear)
        if (idx < 0) {
            const double y0 = baseTable[0].load(std::memory_order_relaxed);
            const double y1 = baseTable[1].load(std::memory_order_relaxed);
            const double slope = y1 - y0;
            return y0 + slope * tableIndex;
        } else {
            const double y0 = baseTable[tableSize - 2].load(std::memory_order_relaxed);
            const double y1 = baseTable[tableSize - 1].load(std::memory_order_relaxed);
            const double slope = y1 - y0;
            return y0 + slope * (tableIndex - (tableSize - 2));
        }
    }

    // Linear interpolation (sufficient for base layer reading)
    const double y0 = baseTable[idx].load(std::memory_order_relaxed);
    const double y1 = baseTable[idx + 1].load(std::memory_order_relaxed);
    return y0 + frac * (y1 - y0);
}

double LayeredTransferFunction::evaluateBaseAndHarmonics(double x) const {
    // CRITICAL: Always evaluate base + harmonics, ignoring spline layer state
    // Used by SplineFitter to read the normalized composite when entering spline mode
    //
    // normalizationScalar is preserved when entering spline mode (not reset to 1.0),
    // so this returns the properly normalized values that SplineFitter needs to fit

    // Interpolate base layer (linear interpolation)
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    const int idx0 = juce::jlimit(0, tableSize - 1, index);
    const int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

    const double y0 = baseTable[idx0].load(std::memory_order_relaxed);
    const double y1 = baseTable[idx1].load(std::memory_order_relaxed);

    const double baseValue = y0 + t * (y1 - y0);

    // Add harmonics
    const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
    const double wtCoeff = coefficients[0];
    const double result = wtCoeff * baseValue + harmonicValue;

    // Apply normalization scalar to match what user sees on screen
    const double normScalar = normalizationScalar.load(std::memory_order_acquire);
    return normScalar * result;
}

double LayeredTransferFunction::evaluateForRendering(double x, double normScalar) const {
    const RenderingMode mode = getRenderingMode();

    switch (mode) {
        case RenderingMode::Spline:
            // Direct spline evaluation (bypasses base + harmonics)
            return splineLayer->evaluate(x);

        case RenderingMode::Paint:
        {
            // Paint mode: Direct base layer output (no normalization)
            // Harmonics are assumed to be baked into base layer
            const double wtCoeff = coefficients[0];
            const double baseValue = getBaseValueAt(x);
            return wtCoeff * baseValue;  // NO NORMALIZATION
        }

        case RenderingMode::Harmonic:
        default:
        {
            // Harmonic mode: Base + harmonics with normalization
            const double wtCoeff = coefficients[0];
            const double baseValue = getBaseValueAt(x);
            const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
            const double unnormalized = wtCoeff * baseValue + harmonicValue;
            return normScalar * unnormalized;  // WITH NORMALIZATION
        }
    }
}

juce::ValueTree LayeredTransferFunction::toValueTree() const {
    juce::ValueTree vt("LayeredTransferFunction");

    // Serialize coefficients
    juce::Array<juce::var> coeffArray;
    for (const double c : coefficients) {
        coeffArray.add(c);
    }
    vt.setProperty("coefficients", coeffArray, nullptr);

    // Serialize base layer
    if (tableSize > 0) {
        juce::ValueTree baseVT("BaseLayer");
        juce::MemoryBlock baseBlob;
        for (int i = 0; i < tableSize; ++i) {
            const double value = baseTable[i].load(std::memory_order_acquire);
            baseBlob.append(&value, sizeof(double));
        }
        baseVT.setProperty("tableData", baseBlob, nullptr);
        vt.addChild(baseVT, -1, nullptr);
    }

    // Serialize harmonic layer (algorithm settings only, no coefficients)
    if (harmonicLayer) {
        vt.addChild(harmonicLayer->toValueTree(), -1, nullptr);
    }

    // NEW: Serialize spline layer
    if (splineLayer) {
        vt.addChild(splineLayer->toValueTree(), -1, nullptr);
    }

    // Serialize normalization scalar
    vt.setProperty("normalizationScalar", normalizationScalar.load(std::memory_order_acquire), nullptr);

    // Serialize normalization enabled state
    vt.setProperty("normalizationEnabled", normalizationEnabled, nullptr);

    // Serialize settings
    vt.setProperty("interpolationMode", static_cast<int>(interpMode), nullptr);
    vt.setProperty("extrapolationMode", static_cast<int>(extrapMode), nullptr);

    return vt;
}

void LayeredTransferFunction::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "LayeredTransferFunction") {
        return;
    }

    // Load coefficients
    // CRITICAL: Always maintain exactly NUM_HARMONIC_COEFFICIENTS (41) coefficients
    // Old presets may have fewer coefficients - pad with zeros if needed
    coefficients.resize(NUM_HARMONIC_COEFFICIENTS, 0.0); // Reset to 41 zeros
    if (vt.hasProperty("coefficients")) {
        const juce::Array<juce::var>* coeffArray = vt.getProperty("coefficients").getArray();
        if (coeffArray != nullptr) {
            const int numToLoad = std::min(coeffArray->size(), NUM_HARMONIC_COEFFICIENTS);
            for (int i = 0; i < numToLoad; ++i) {
                coefficients[i] = static_cast<double>((*coeffArray)[i]);
            }
        }
    }

    // Load base layer
    const auto baseVT = vt.getChildWithName("BaseLayer");
    if (baseVT.isValid() && baseVT.hasProperty("tableData")) {
        const juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
        const double* data = static_cast<const double*>(baseBlob.getData());
        const int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));

        for (int i = 0; i < std::min(numValues, tableSize); ++i) {
            baseTable[i].store(data[i], std::memory_order_release);
        }
    }

    // Load harmonic layer (algorithm settings only)
    const auto harmonicVT = vt.getChildWithName("HarmonicLayer");
    if (harmonicVT.isValid()) {
        harmonicLayer->fromValueTree(harmonicVT);
        harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);
    }

    // NEW: Load spline layer
    const auto splineVT = vt.getChildWithName("SplineLayer");
    if (splineVT.isValid()) {
        splineLayer->fromValueTree(splineVT);
    }

    // Load normalization scalar (optional - will be recomputed anyway)
    if (vt.hasProperty("normalizationScalar")) {
        normalizationScalar.store(static_cast<double>(vt.getProperty("normalizationScalar")),
                                  std::memory_order_release);
    }

    // Load normalization enabled state (default to true if not present for backward compatibility)
    if (vt.hasProperty("normalizationEnabled")) {
        normalizationEnabled = static_cast<bool>(vt.getProperty("normalizationEnabled"));
    } else {
        normalizationEnabled = true; // Safe default for old presets
    }

    // Load settings
    if (vt.hasProperty("interpolationMode")) {
        interpMode = static_cast<InterpolationMode>(static_cast<int>(vt.getProperty("interpolationMode")));
    }
    if (vt.hasProperty("extrapolationMode")) {
        extrapMode = static_cast<ExtrapolationMode>(static_cast<int>(vt.getProperty("extrapolationMode")));
    }

    // Increment version to trigger LUT render on preset load
    versionCounter.fetch_add(1, std::memory_order_release);
}

} // namespace dsp_core
