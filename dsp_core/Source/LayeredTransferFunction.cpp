#include "LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

LayeredTransferFunction::LayeredTransferFunction(int size, double minVal, double maxVal)
    : tableSize(size), minValue(minVal), maxValue(maxVal), baseTable(size), compositeTable(size) {

    // Initialize base layer to identity: y = x
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        baseTable[i].store(x);
    }

    // Create harmonic layer with default coefficients (WT=1.0, harmonics=0)
    harmonicLayer = std::make_unique<HarmonicLayer>(19);

    // Precompute harmonic basis functions
    harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);

    // Initial composite = base (since harmonics are zero)
    updateComposite();
}

double LayeredTransferFunction::getBaseLayerValue(int index) const {
    if (index >= 0 && index < tableSize) {
        return baseTable[index].load();
    }
    return 0.0;
}

void LayeredTransferFunction::setBaseLayerValue(int index, double value) {
    if (index >= 0 && index < tableSize) {
        baseTable[index].store(value);
    }
}

void LayeredTransferFunction::clearBaseLayer() {
    for (int i = 0; i < tableSize; ++i) {
        baseTable[i].store(0.0);
    }
}

HarmonicLayer& LayeredTransferFunction::getHarmonicLayer() {
    return *harmonicLayer;
}

const HarmonicLayer& LayeredTransferFunction::getHarmonicLayer() const {
    return *harmonicLayer;
}

double LayeredTransferFunction::getCompositeValue(int index) const {
    if (index >= 0 && index < tableSize) {
        return compositeTable[index].load();
    }
    return 0.0;
}

void LayeredTransferFunction::updateComposite() {
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);

        // Evaluate layers
        double baseValue = baseTable[i].load();
        double harmonicValue = harmonicLayer->evaluate(x, tableSize);
        double wtMix = harmonicLayer->getCoefficient(0);

        // Mix: WT coefficient × base + harmonics
        double composite = wtMix * baseValue + harmonicValue;

        // Store (no soft-clipping yet - preserve original behavior initially)
        compositeTable[i].store(composite);
    }

    // DISABLED: normalizeByMaximum() breaks differential solving in drawing mode!
    // The user draws at position Y, but normalization scales the entire composite table,
    // so the curve appears at Y * scaleFactor (which changes every stroke).
    // This invalidates: Base = (Composite - Harmonics) / WT
    // TODO: Implement soft-clipping or per-sample limiting instead
    // normalizeByMaximum();
}

double LayeredTransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize)
        return 0.0;
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minValue, maxValue);
}

void LayeredTransferFunction::normalizeByMaximum() {
    double maxAbsValue = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        double absValue = std::abs(compositeTable[i].load());
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }

    if (maxAbsValue > 0.0) {
        double scale = 1.0 / maxAbsValue;
        for (int i = 0; i < tableSize; ++i) {
            compositeTable[i].store(compositeTable[i].load() * scale);
        }
    }
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
            return applyTransferFunctionLinear(x);
        case InterpolationMode::Cubic:
            return applyTransferFunctionCubic(x);
        case InterpolationMode::CatmullRom:
        default:
            return applyTransferFunctionCatmullRom(x);
    }
}

// Interpolation methods adapted from TransferFunction
double LayeredTransferFunction::applyTransferFunctionLinear(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index);
    double y1 = getSample(index + 1);
    return interpolateLinear(y0, y1, t);
}

double LayeredTransferFunction::applyTransferFunctionCubic(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);
    return interpolateCubic(y0, y1, y2, y3, t);
}

double LayeredTransferFunction::applyTransferFunctionCatmullRom(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);
    return interpolateCatmullRom(y0, y1, y2, y3, t);
}

double LayeredTransferFunction::interpolateLinear(double y0, double y1, double t) const {
    return y0 + t * (y1 - y0);
}

double LayeredTransferFunction::interpolateCubic(double y0, double y1, double y2, double y3, double t) const {
    double a0 = y3 - y2 - y0 + y1;
    double a1 = y0 - y1 - a0;
    double a2 = y2 - y0;
    double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double LayeredTransferFunction::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) const {
    return 0.5 * ((2.0 * y1) +
                  (-y0 + y2) * t +
                  (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
}

double LayeredTransferFunction::getSample(int i) const {
    bool isBelow = (i < 0);
    bool isAbove = (i > tableSize - 1);
    if (isBelow && (extrapMode == ExtrapolationMode::Linear)) {
        double slope = compositeTable[1].load() - compositeTable[0].load();
        return compositeTable[0].load() + slope * i;
    }
    else if (isAbove && extrapMode == ExtrapolationMode::Linear) {
        double slope = compositeTable[tableSize - 1].load() - compositeTable[tableSize - 2].load();
        return compositeTable[tableSize - 1].load() + slope * (i - tableSize + 1);
    }
    else {
        int clampedIdx = juce::jlimit(0, tableSize - 1, i);
        return compositeTable[clampedIdx].load();
    }
}

juce::ValueTree LayeredTransferFunction::toValueTree() const {
    juce::ValueTree vt("LayeredTransferFunction");

    // Serialize base layer
    juce::ValueTree baseVT("BaseLayer");
    juce::MemoryBlock baseBlob;
    for (int i = 0; i < tableSize; ++i) {
        double value = baseTable[i].load();
        baseBlob.append(&value, sizeof(double));
    }
    baseVT.setProperty("tableData", baseBlob, nullptr);
    vt.addChild(baseVT, -1, nullptr);

    // Serialize harmonic layer
    vt.addChild(harmonicLayer->toValueTree(), -1, nullptr);

    // Serialize settings
    vt.setProperty("interpolationMode", static_cast<int>(interpMode), nullptr);
    vt.setProperty("extrapolationMode", static_cast<int>(extrapMode), nullptr);

    return vt;
}

void LayeredTransferFunction::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "LayeredTransferFunction") {
        return;
    }

    // Load base layer
    auto baseVT = vt.getChildWithName("BaseLayer");
    if (baseVT.isValid() && baseVT.hasProperty("tableData")) {
        juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
        const double* data = static_cast<const double*>(baseBlob.getData());
        int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));

        for (int i = 0; i < std::min(numValues, tableSize); ++i) {
            baseTable[i].store(data[i]);
        }
    }

    // Load harmonic layer
    auto harmonicVT = vt.getChildWithName("HarmonicLayer");
    if (harmonicVT.isValid()) {
        harmonicLayer->fromValueTree(harmonicVT);
        harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);
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

//==============================================================================
// Compatibility methods for TransferFunction API
//==============================================================================

void LayeredTransferFunction::setTableValue(int index, double value) {
    // Set the BASE layer value (user drawing operates on base)
    // NOTE: Does NOT call updateComposite() - caller must do so after batch edits
    // This prevents recomputing the entire table on every paint stroke point
    setBaseLayerValue(index, value);
}

void LayeredTransferFunction::applyFunction(std::function<double(double)> func) {
    // Apply function to BASE layer
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        double y = func(x);
        setBaseLayerValue(i, y);
    }
    // Update composite after modifying base
    updateComposite();
}

void LayeredTransferFunction::writeToStream(juce::OutputStream& stream) const {
    auto vt = toValueTree();
    vt.writeToStream(stream);
}

void LayeredTransferFunction::readFromStream(juce::InputStream& stream) {
    auto vt = juce::ValueTree::readFromStream(stream);
    fromValueTree(vt);
}

} // namespace dsp_core
