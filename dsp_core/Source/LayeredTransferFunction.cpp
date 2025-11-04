#include "LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>

// Branch prediction hints (improve CPU branch predictor performance)
// GCC/Clang support __builtin_expect for optimization
#if defined(__GNUC__) || defined(__clang__)
  #define juce_likely(x)    __builtin_expect(!!(x), 1)
  #define juce_unlikely(x)  __builtin_expect(!!(x), 0)
#else
  #define juce_likely(x)    (x)
  #define juce_unlikely(x)  (x)
#endif

namespace dsp_core {

LayeredTransferFunction::LayeredTransferFunction(int size, double minVal, double maxVal)
    : tableSize(size), minValue(minVal), maxValue(maxVal),
      harmonicLayer(std::make_unique<HarmonicLayer>(40)),
      splineLayer(std::make_unique<SplineLayer>()),  // NEW: Initialize spline layer
      coefficients(41, 0.0),  // 41 coefficients: [0] = WT, [1..40] = harmonics
      baseTable(size),
      compositeTable(size) {

    // Pre-allocate scratch buffer for updateComposite() (eliminates heap allocation)
    unnormalizedMixBuffer.resize(size);

    // Initialize coefficients
    coefficients[0] = 1.0;  // Default WT mix = 1.0 (full base layer)
    // coefficients[1..40] already initialized to 0.0

    // Initialize base layer to identity: y = x
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        baseTable[i].store(x, std::memory_order_release);
    }

    // Precompute harmonic basis functions
    harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);

    // Initial composite = base (since harmonics are zero)
    updateComposite();
}

double LayeredTransferFunction::getBaseLayerValue(int index) const {
    if (index >= 0 && index < tableSize) {
        return baseTable[index].load(std::memory_order_acquire);
    }
    return 0.0;
}

void LayeredTransferFunction::setBaseLayerValue(int index, double value) {
    if (index >= 0 && index < tableSize) {
        baseTable[index].store(value, std::memory_order_release);

        // NEW: Invalidate cache on base layer changes
        invalidateCompositeCache();
    }
}

void LayeredTransferFunction::clearBaseLayer() {
    for (int i = 0; i < tableSize; ++i) {
        baseTable[i].store(0.0, std::memory_order_release);
    }
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

void LayeredTransferFunction::setCoefficient(int index, double value) {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        coefficients[index] = value;

        // NEW: Invalidate cache on coefficient changes
        invalidateCompositeCache();
    }
}

double LayeredTransferFunction::getCoefficient(int index) const {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        return coefficients[index];
    }
    return 0.0;
}

double LayeredTransferFunction::getCompositeValue(int index) const {
    if (index >= 0 && index < tableSize) {
        return compositeTable[index].load(std::memory_order_acquire);
    }
    return 0.0;
}

void LayeredTransferFunction::updateComposite() {
    if (splineLayerEnabled.load(std::memory_order_acquire)) {
        updateCompositeSplineMode();
    } else {
        updateCompositeHarmonicMode();
    }

    // Mark cache as valid after rebuild
    compositeCacheValid.store(true, std::memory_order_release);
}

void LayeredTransferFunction::updateCompositeSplineMode() {
    // Spline mode: Cache spline evaluation results
    // No normalization (normScalar locked to 1.0 in spline mode)
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        double value = splineLayer->evaluate(x);
        compositeTable[i].store(value, std::memory_order_release);
    }
}

void LayeredTransferFunction::updateCompositeHarmonicMode() {
    // Ensure buffer is correct size (handles edge case of table resize)
    if (static_cast<int>(unnormalizedMixBuffer.size()) != tableSize) {
        unnormalizedMixBuffer.resize(tableSize);
    }

    // Step 1: Compute unnormalized mix and find max absolute value
    double maxAbsValue = 0.0;

    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);

        // Get layer values (NEVER modified by this function)
        double baseValue = baseTable[i].load(std::memory_order_acquire);
        double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
        double wavetableCoeff = coefficients[0];  // WT mix coefficient

        // Compute unnormalized mix: UnNorm = wtCoeff*Base + HarmonicSum
        // Note: harmonicLayer->evaluate() already sums harmonicCoeff[n] * Harmonic_n(x)
        double unnormalized = wavetableCoeff * baseValue + harmonicValue;

        unnormalizedMixBuffer[i] = unnormalized;

        // Track maximum absolute value
        double absValue = std::abs(unnormalized);
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }

    // Step 2: Compute normalization scalar (or use frozen/disabled scalar)
    double normScalar = computeNormalizationScalar(maxAbsValue);

    // Step 3: Store normalized composite
    for (int i = 0; i < tableSize; ++i) {
        double normalized = normScalar * unnormalizedMixBuffer[i];
        compositeTable[i].store(normalized, std::memory_order_release);
    }
}

double LayeredTransferFunction::computeNormalizationScalar(double maxAbsValue) {
    if (!normalizationEnabled) {
        // Normalization disabled: bypass scaling (allow values > ±1.0)
        normalizationScalar.store(1.0, std::memory_order_release);
        return 1.0;
    }

    if (!deferNormalization) {
        // Normal mode: recalculate normalization scalar
        double normScalar = 1.0;
        if (maxAbsValue > 1e-12) {  // Avoid division by zero
            normScalar = 1.0 / maxAbsValue;
        }
        normalizationScalar.store(normScalar, std::memory_order_release);
        return normScalar;
    }

    // Deferred mode: keep using existing normScalar
    return normalizationScalar.load(std::memory_order_acquire);
}

void LayeredTransferFunction::setDeferNormalization(bool shouldDefer) {
    deferNormalization = shouldDefer;

    // When exiting deferred mode, immediately recalculate normalization
    if (!shouldDefer) {
        updateComposite();
    }
}

bool LayeredTransferFunction::isNormalizationDeferred() const {
    return deferNormalization;
}

void LayeredTransferFunction::setNormalizationEnabled(bool enabled) {
    normalizationEnabled = enabled;

    // Immediately update composite to apply new normalization state
    // This ensures the change takes effect without waiting for next model mutation
    updateComposite();
}

bool LayeredTransferFunction::isNormalizationEnabled() const {
    return normalizationEnabled;
}

void LayeredTransferFunction::setSplineLayerEnabled(bool enabled) {
    if (enabled) {
        // Entering spline mode:
        // - Harmonics are kept intact (hidden but present for undo)
        // - Spline fits to composite (base + harmonics)
        // - Lock normalization to identity (user controls amplitude via anchors)
        normalizationScalar.store(1.0, std::memory_order_release);
    }

    splineLayerEnabled.store(enabled, std::memory_order_release);
    invalidateCompositeCache();
}

bool LayeredTransferFunction::hasNonZeroHarmonics() const {
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        if (std::abs(coefficients[i]) > 1e-9) {
            return true;
        }
    }
    return false;
}

bool LayeredTransferFunction::isSplineLayerEnabled() const {
    return splineLayerEnabled.load(std::memory_order_acquire);
}

void LayeredTransferFunction::invalidateCompositeCache() {
    compositeCacheValid.store(false, std::memory_order_release);
}

bool LayeredTransferFunction::isCompositeCacheValid() const {
    return compositeCacheValid.load(std::memory_order_acquire);
}

double LayeredTransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize)
        return 0.0;
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minValue, maxValue);
}

double LayeredTransferFunction::applyTransferFunction(double x) const {
    // Check cache validity (fast path ~95%)
    if (juce_likely(compositeCacheValid.load(std::memory_order_acquire))) {
        return interpolate(x);  // Use cached composite table (~5-10ns)
    }

    // Slow path: direct evaluation (~40-50ns)
    return evaluateDirect(x);
}

void LayeredTransferFunction::processBlock(double* samples, int numSamples) {
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
    double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    // Avoids lambda overhead, branch prediction issues, and extra atomic loads
    if (juce_likely (extrapMode == ExtrapolationMode::Clamp)) {
        int idx0 = juce::jlimit(0, tableSize - 1, index);
        int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

        double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        double y1 = compositeTable[idx1].load(std::memory_order_relaxed);

        return y0 + t * (y1 - y0);
    }

    // Linear extrapolation path (requires boundary checks and slope calculations)
    double y0, y1;

    // Handle index
    if (index < 0) {
        double slope = compositeTable[1].load(std::memory_order_acquire) -
                      compositeTable[0].load(std::memory_order_acquire);
        y0 = compositeTable[0].load(std::memory_order_acquire) + slope * index;
    } else if (index >= tableSize) {
        double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                      compositeTable[tableSize - 2].load(std::memory_order_acquire);
        y0 = compositeTable[tableSize - 1].load(std::memory_order_acquire) +
             slope * (index - tableSize + 1);
    } else {
        y0 = compositeTable[index].load(std::memory_order_acquire);
    }

    // Handle index + 1
    int index1 = index + 1;
    if (index1 < 0) {
        double slope = compositeTable[1].load(std::memory_order_acquire) -
                      compositeTable[0].load(std::memory_order_acquire);
        y1 = compositeTable[0].load(std::memory_order_acquire) + slope * index1;
    } else if (index1 >= tableSize) {
        double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                      compositeTable[tableSize - 2].load(std::memory_order_acquire);
        y1 = compositeTable[tableSize - 1].load(std::memory_order_acquire) +
             slope * (index1 - tableSize + 1);
    } else {
        y1 = compositeTable[index1].load(std::memory_order_acquire);
    }

    return y0 + t * (y1 - y0);
}

double LayeredTransferFunction::interpolateCubic(double x) const {
    // Map x from signal range to table index
    double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    if (juce_likely (extrapMode == ExtrapolationMode::Clamp)) {
        int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        int idx1 = juce::jlimit(0, tableSize - 1, index);
        int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        double y1 = compositeTable[idx1].load(std::memory_order_relaxed);
        double y2 = compositeTable[idx2].load(std::memory_order_relaxed);
        double y3 = compositeTable[idx3].load(std::memory_order_relaxed);

        double a0 = y3 - y2 - y0 + y1;
        double a1 = y0 - y1 - a0;
        double a2 = y2 - y0;
        double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            double slope = compositeTable[1].load(std::memory_order_acquire) -
                          compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        } else if (i >= tableSize) {
            double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                          compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) +
                   slope * (i - tableSize + 1);
        } else {
            return compositeTable[i].load(std::memory_order_acquire);
        }
    };

    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);

    double a0 = y3 - y2 - y0 + y1;
    double a1 = y0 - y1 - a0;
    double a2 = y2 - y0;
    double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double LayeredTransferFunction::interpolateCatmullRom(double x) const {
    // Map x from signal range to table index
    double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    if (juce_likely (extrapMode == ExtrapolationMode::Clamp)) {
        int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        int idx1 = juce::jlimit(0, tableSize - 1, index);
        int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        double y1 = compositeTable[idx1].load(std::memory_order_relaxed);
        double y2 = compositeTable[idx2].load(std::memory_order_relaxed);
        double y3 = compositeTable[idx3].load(std::memory_order_relaxed);

        return 0.5 * ((2.0 * y1) +
                      (-y0 + y2) * t +
                      (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            double slope = compositeTable[1].load(std::memory_order_acquire) -
                          compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        } else if (i >= tableSize) {
            double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                          compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) +
                   slope * (i - tableSize + 1);
        } else {
            return compositeTable[i].load(std::memory_order_acquire);
        }
    };

    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);

    return 0.5 * ((2.0 * y1) +
                  (-y0 + y2) * t +
                  (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
}

double LayeredTransferFunction::evaluateDirect(double x) const {
    bool splineEnabled = splineLayerEnabled.load(std::memory_order_acquire);

    if (splineEnabled) {
        // Spline mode: direct PCHIP evaluation, no normalization
        // User has direct amplitude control via anchor placement
        return splineLayer->evaluate(x);
    } else {
        // Harmonic mode: base + harmonics with normalization
        double baseValue = interpolateBase(x);  // NEW helper method
        double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
        double wtCoeff = coefficients[0];
        double result = wtCoeff * baseValue + harmonicValue;

        // Apply normalization scalar (only in harmonic mode)
        double normScalar = normalizationScalar.load(std::memory_order_acquire);
        return normScalar * result;
    }
}

double LayeredTransferFunction::interpolateBase(double x) const {
    // Map x from signal range to table index
    double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;

    // Use linear interpolation (simple and fast for direct path)
    int idx0 = juce::jlimit(0, tableSize - 1, index);
    int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

    double y0 = baseTable[idx0].load(std::memory_order_relaxed);
    double y1 = baseTable[idx1].load(std::memory_order_relaxed);

    return y0 + t * (y1 - y0);
}

juce::ValueTree LayeredTransferFunction::toValueTree() const {
    juce::ValueTree vt("LayeredTransferFunction");

    // Serialize coefficients
    juce::Array<juce::var> coeffArray;
    for (double c : coefficients) {
        coeffArray.add(c);
    }
    vt.setProperty("coefficients", coeffArray, nullptr);

    // Serialize base layer
    if (tableSize > 0) {
        juce::ValueTree baseVT("BaseLayer");
        juce::MemoryBlock baseBlob;
        for (int i = 0; i < tableSize; ++i) {
            double value = baseTable[i].load(std::memory_order_acquire);
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
    if (vt.hasProperty("coefficients")) {
        juce::Array<juce::var>* coeffArray = vt.getProperty("coefficients").getArray();
        if (coeffArray != nullptr) {
            coefficients.clear();
            for (const auto& var : *coeffArray) {
                coefficients.push_back(static_cast<double>(var));
            }
        }
    }

    // Load base layer
    auto baseVT = vt.getChildWithName("BaseLayer");
    if (baseVT.isValid() && baseVT.hasProperty("tableData")) {
        juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
        const double* data = static_cast<const double*>(baseBlob.getData());
        int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));

        for (int i = 0; i < std::min(numValues, tableSize); ++i) {
            baseTable[i].store(data[i], std::memory_order_release);
        }
    }

    // Load harmonic layer (algorithm settings only)
    auto harmonicVT = vt.getChildWithName("HarmonicLayer");
    if (harmonicVT.isValid()) {
        harmonicLayer->fromValueTree(harmonicVT);
        harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);
    }

    // NEW: Load spline layer
    auto splineVT = vt.getChildWithName("SplineLayer");
    if (splineVT.isValid()) {
        splineLayer->fromValueTree(splineVT);
    }

    // Load normalization scalar (optional - will be recomputed anyway)
    if (vt.hasProperty("normalizationScalar")) {
        normalizationScalar.store(static_cast<double>(vt.getProperty("normalizationScalar")), std::memory_order_release);
    }

    // Load normalization enabled state (default to true if not present for backward compatibility)
    if (vt.hasProperty("normalizationEnabled")) {
        normalizationEnabled = static_cast<bool>(vt.getProperty("normalizationEnabled"));
    } else {
        normalizationEnabled = true;  // Safe default for old presets
    }

    // Load settings
    if (vt.hasProperty("interpolationMode")) {
        interpMode = static_cast<InterpolationMode>(static_cast<int>(vt.getProperty("interpolationMode")));
    }
    if (vt.hasProperty("extrapolationMode")) {
        extrapMode = static_cast<ExtrapolationMode>(static_cast<int>(vt.getProperty("extrapolationMode")));
    }

    // Recompute composite
    updateComposite();
}

} // namespace dsp_core
