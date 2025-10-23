#include "LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>
#include <juce_data_structures/juce_data_structures.h>

namespace dsp_core {

LayeredTransferFunction::LayeredTransferFunction(int size, double minVal, double maxVal)
    : tableSize(size), minValue(minVal), maxValue(maxVal),
      harmonicLayer(std::make_unique<HarmonicLayer>(19)),
      coefficients(20, 0.0),  // 20 coefficients: [0] = WT, [1..19] = harmonics
      baseTable(size),
      compositeTable(size) {

    // Initialize coefficients
    coefficients[0] = 1.0;  // Default WT mix = 1.0 (full base layer)
    // coefficients[1..19] already initialized to 0.0

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

void LayeredTransferFunction::setCoefficient(int index, double value) {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        coefficients[index] = value;
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
    // Step 1: Compute unnormalized mix and find max absolute value
    std::vector<double> unnormalizedMix(tableSize);
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

        unnormalizedMix[i] = unnormalized;

        // Track maximum absolute value
        double absValue = std::abs(unnormalized);
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }

    // Step 2: Compute normalization scalar (or use frozen scalar if deferred)
    double normScalar = normalizationScalar.load(std::memory_order_acquire);  // Default to existing scalar

    if (!deferNormalization) {
        // Normal mode: recalculate normalization scalar
        normScalar = 1.0;
        if (maxAbsValue > 1e-12) {  // Avoid division by zero
            normScalar = 1.0 / maxAbsValue;
        }
        normalizationScalar.store(normScalar, std::memory_order_release);
    }
    // else: deferred mode - keep using existing normScalar

    // Step 3: Store normalized composite
    for (int i = 0; i < tableSize; ++i) {
        double normalized = normScalar * unnormalizedMix[i];
        compositeTable[i].store(normalized, std::memory_order_release);
    }
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

double LayeredTransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize)
        return 0.0;
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minValue, maxValue);
}

double LayeredTransferFunction::applyTransferFunction(double x) const {
    return interpolate(x);
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

    // Get samples with extrapolation handling
    auto getSample = [this](int i) -> double {
        bool isBelow = (i < 0);
        bool isAbove = (i > tableSize - 1);

        if (isBelow && (extrapMode == ExtrapolationMode::Linear)) {
            double slope = compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        }
        else if (isAbove && extrapMode == ExtrapolationMode::Linear) {
            double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) - compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (i - tableSize + 1);
        }
        else {
            int clampedIdx = juce::jlimit(0, tableSize - 1, i);
            return compositeTable[clampedIdx].load(std::memory_order_acquire);
        }
    };

    double y0 = getSample(index);
    double y1 = getSample(index + 1);

    return y0 + t * (y1 - y0);
}

double LayeredTransferFunction::interpolateCubic(double x) const {
    // Map x from signal range to table index
    double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;

    // Get samples with extrapolation handling
    auto getSample = [this](int i) -> double {
        bool isBelow = (i < 0);
        bool isAbove = (i > tableSize - 1);

        if (isBelow && (extrapMode == ExtrapolationMode::Linear)) {
            double slope = compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        }
        else if (isAbove && extrapMode == ExtrapolationMode::Linear) {
            double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) - compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (i - tableSize + 1);
        }
        else {
            int clampedIdx = juce::jlimit(0, tableSize - 1, i);
            return compositeTable[clampedIdx].load(std::memory_order_acquire);
        }
    };

    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);

    // Cubic interpolation (from TransferFunction)
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

    // Get samples with extrapolation handling
    auto getSample = [this](int i) -> double {
        bool isBelow = (i < 0);
        bool isAbove = (i > tableSize - 1);

        if (isBelow && (extrapMode == ExtrapolationMode::Linear)) {
            double slope = compositeTable[1].load(std::memory_order_acquire) - compositeTable[0].load(std::memory_order_acquire);
            return compositeTable[0].load(std::memory_order_acquire) + slope * i;
        }
        else if (isAbove && extrapMode == ExtrapolationMode::Linear) {
            double slope = compositeTable[tableSize - 1].load(std::memory_order_acquire) - compositeTable[tableSize - 2].load(std::memory_order_acquire);
            return compositeTable[tableSize - 1].load(std::memory_order_acquire) + slope * (i - tableSize + 1);
        }
        else {
            int clampedIdx = juce::jlimit(0, tableSize - 1, i);
            return compositeTable[clampedIdx].load(std::memory_order_acquire);
        }
    };

    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);

    // Catmull-Rom interpolation (from TransferFunction)
    return 0.5 * ((2.0 * y1) +
                  (-y0 + y2) * t +
                  (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
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
    } else {
        // Log or handle invalid table size if necessary
        jassertfalse; // Debug assertion for invalid table size
    }

    // Serialize harmonic layer (algorithm settings only, no coefficients)
    if (harmonicLayer) {
        vt.addChild(harmonicLayer->toValueTree(), -1, nullptr);
    } else {
        // Log or handle null harmonic layer if necessary
        jassertfalse; // Debug assertion for null harmonic layer
    }

    // Serialize normalization scalar
    vt.setProperty("normalizationScalar", normalizationScalar.load(std::memory_order_acquire), nullptr);

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

    // Load normalization scalar (optional - will be recomputed anyway)
    if (vt.hasProperty("normalizationScalar")) {
        normalizationScalar.store(static_cast<double>(vt.getProperty("normalizationScalar")), std::memory_order_release);
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
