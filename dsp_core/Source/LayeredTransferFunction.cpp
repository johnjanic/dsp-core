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
      baseTable(tableSize), compositeTable(tableSize) {

    DBG("[MODEL:" + juce::String(instanceId) + "] LayeredTransferFunction instance created");

    // Pre-allocate scratch buffer for updateComposite() (eliminates heap allocation)
    unnormalizedMixBuffer.resize(tableSize);

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

    // Initial composite = base (since harmonics are zero)
    updateComposite();
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
}

void LayeredTransferFunction::updateCompositeSplineMode() {
    // Spline mode: Cache spline evaluation results
    // No normalization (normScalar locked to 1.0 in spline mode)
    for (int i = 0; i < tableSize; ++i) {
        const double x = normalizeIndex(i);
        const double value = splineLayer->evaluate(x);
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
        const double x = normalizeIndex(i);

        // Get layer values (NEVER modified by this function)
        const double baseValue = baseTable[i].load(std::memory_order_acquire);
        const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
        const double wavetableCoeff = coefficients[0]; // WT mix coefficient

        // Compute unnormalized mix: UnNorm = wtCoeff*Base + HarmonicSum
        // Note: harmonicLayer->evaluate() already sums harmonicCoeff[n] * Harmonic_n(x)
        const double unnormalized = wavetableCoeff * baseValue + harmonicValue;

        unnormalizedMixBuffer[i] = unnormalized;

        // Track maximum absolute value
        const double absValue = std::abs(unnormalized);
        maxAbsValue = std::max(absValue, maxAbsValue);
    }

    // Step 2: Compute normalization scalar (or use frozen/disabled scalar)
    const double normScalar = computeNormalizationScalar(maxAbsValue);

    // Step 3: Store normalized composite
    for (int i = 0; i < tableSize; ++i) {
        const double normalized = normScalar * unnormalizedMixBuffer[i];
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
        if (maxAbsValue > kNormalizationEpsilon) { // Avoid division by zero
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
        const uint64_t newVersion = versionCounter.fetch_add(1, std::memory_order_release) + 1;
        DBG("[MODEL:" + juce::String(instanceId) + "] Paint stroke complete, version incremented to " + juce::String(static_cast<int64_t>(newVersion)));
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
    versionCounter.fetch_add(1, std::memory_order_release);
}

bool LayeredTransferFunction::isNormalizationEnabled() const {
    return normalizationEnabled;
}

void LayeredTransferFunction::setSplineLayerEnabled(bool enabled) {
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

    splineLayerEnabled.store(enabled, std::memory_order_release);
    versionCounter.fetch_add(1, std::memory_order_release);
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

    // Capture composite curve (base + harmonics with normalization applied)
    // This preserves the visual appearance of the curve exactly
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = getCompositeValue(i);
        setBaseLayerValue(i, compositeValue);
    }

    // Set WT coefficient to 1.0 (CRITICAL: enables the baked base layer)
    // If WT was 0 during harmonic editing, the baked base layer would be invisible
    coefficients[0] = 1.0;

    // Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Regenerate composite (now just base layer with WT mix = 1.0)
    // Note: normalizationScalar is preserved (not reset to 1.0)
    updateComposite();
    versionCounter.fetch_add(1, std::memory_order_release);

    return true;
}

void LayeredTransferFunction::bakeCompositeToBase() {
    // Capture composite curve (base + harmonics with normalization applied)
    // This preserves the visual appearance of the curve exactly
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = getCompositeValue(i);
        setBaseLayerValue(i, compositeValue);
    }

    // Set WT coefficient to 1.0 (enable base layer)
    coefficients[0] = 1.0;

    // Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Regenerate composite (now just base layer with WT mix = 1.0)
    updateComposite();
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

    updateComposite();
    versionCounter.fetch_add(1, std::memory_order_release);
}

bool LayeredTransferFunction::isSplineLayerEnabled() const {
    return splineLayerEnabled.load(std::memory_order_acquire);
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
    // Avoids lambda overhead, branch prediction issues, and extra atomic loads
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index);
        const int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

        const double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        const double y1 = compositeTable[idx1].load(std::memory_order_relaxed);

        return y0 + t * (y1 - y0);
    }

    // Linear extrapolation path (requires boundary checks and slope calculations)
    double y0;
    double y1;

    // Handle index
    if (index < 0) {
        const double slope =
            compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
        y0 = compositeTable[0].load(std::memory_order_acquire) + slope * index;
    } else if (index >= tableSize) {
        const double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                       compositeTable[tableSize - 2].load(std::memory_order_acquire);
        y0 = compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (index - tableSize + 1);
    } else {
        y0 = compositeTable[index].load(std::memory_order_acquire);
    }

    // Handle index + 1
    const int index1 = index + 1;
    if (index1 < 0) {
        const double slope =
            compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
        y1 = compositeTable[0].load(std::memory_order_acquire) + slope * index1;
    } else if (index1 >= tableSize) {
        const double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                       compositeTable[tableSize - 2].load(std::memory_order_acquire);
        y1 = compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (index1 - tableSize + 1);
    } else {
        y1 = compositeTable[index1].load(std::memory_order_acquire);
    }

    return y0 + t * (y1 - y0);
}

double LayeredTransferFunction::interpolateCubic(double x) const {
    // Map x from signal range to table index
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        const int idx1 = juce::jlimit(0, tableSize - 1, index);
        const int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        const int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        const double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        const double y1 = compositeTable[idx1].load(std::memory_order_relaxed);
        const double y2 = compositeTable[idx2].load(std::memory_order_relaxed);
        const double y3 = compositeTable[idx3].load(std::memory_order_relaxed);

        const double a0 = y3 - y2 - y0 + y1;
        const double a1 = y0 - y1 - a0;
        const double a2 = y2 - y0;
        const double a3 = y1;
        return a0 * t * t * t + a1 * t * t + a2 * t + a3;
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            const double slope =
                compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        }
        if (i >= tableSize) {
            const double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                           compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (i - tableSize + 1);
        }
        return compositeTable[i].load(std::memory_order_acquire);
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
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        const int idx1 = juce::jlimit(0, tableSize - 1, index);
        const int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        const int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        const double y0 = compositeTable[idx0].load(std::memory_order_relaxed);
        const double y1 = compositeTable[idx1].load(std::memory_order_relaxed);
        const double y2 = compositeTable[idx2].load(std::memory_order_relaxed);
        const double y3 = compositeTable[idx3].load(std::memory_order_relaxed);

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            const double slope =
                compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        }
        if (i >= tableSize) {
            const double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) -
                           compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (i - tableSize + 1);
        }
        return compositeTable[i].load(std::memory_order_acquire);
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

    // Recompute composite
    updateComposite();

    // Increment version to trigger LUT render on preset load
    versionCounter.fetch_add(1, std::memory_order_release);
}

} // namespace dsp_core
