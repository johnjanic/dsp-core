#include "TransferFunction.h"

namespace dsp_core {

TransferFunction::TransferFunction(int size, double minVal, double maxVal)
    : tableSize(size), minSignalValue(minVal), maxSignalValue(maxVal), table(size) {
    reset();
}

void TransferFunction::reset() {
    for (int i = 0; i < tableSize; ++i)
        setTableValue(i, normalizeIndex(i));
}

double TransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize)
        return 0.0;
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minSignalValue,
                      maxSignalValue);
}

void TransferFunction::setInterpolationMode(InterpolationMode mode) {
    interpolationMode = mode;
}

void TransferFunction::setExtrapolationMode(ExtrapolationMode mode) {
    extrapolationMode = mode;
}

void TransferFunction::setTableValue(int index, double value) {
    if (index >= 0 && index < tableSize)
        table[index].store(value);
}

void TransferFunction::setTableValues(const double* samples, const int numSamples) {
    for (int i = 0; i < numSamples && i < tableSize; ++i) {
        setTableValue(i, samples[i]);
    }
}

double TransferFunction::getTableValue(int index) const {
    if (index >= 0 && index < tableSize)
        return table[index].load();
    return 0.0;
}

double TransferFunction::applyTransferFunction(double x) const {
    switch (interpolationMode) {
    case InterpolationMode::Linear:
        return applyTransferFunctionLinear(x);
    case InterpolationMode::Cubic:
        return applyTransferFunctionCubic(x);
    case InterpolationMode::CatmullRom:
    default:
        return applyTransferFunctionCatmullRom(x);
    }
}

double TransferFunction::applyTransferFunctionLinear(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index);
    double y1 = getSample(index);
    return interpolateLinear(y0, y1, t);
}

double TransferFunction::applyTransferFunctionCubic(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);
    return interpolateCubic(y0, y1, y2, y3, t);
}

double TransferFunction::applyTransferFunctionCatmullRom(double x) const {
    double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    int index = static_cast<int>(x_proj);
    double t = x_proj - index;
    double y0 = getSample(index - 1);
    double y1 = getSample(index);
    double y2 = getSample(index + 1);
    double y3 = getSample(index + 2);
    return interpolateCatmullRom(y0, y1, y2, y3, t);
}

double TransferFunction::interpolateLinear(double y0, double y1, double t) const {
    return y0 + t * (y1 - y0);
}

double TransferFunction::interpolateCubic(double y0, double y1, double y2, double y3, double t) const {
    double a0 = y3 - y2 - y0 + y1;
    double a1 = y0 - y1 - a0;
    double a2 = y2 - y0;
    double a3 = y1;
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

double TransferFunction::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) const {
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
}

double TransferFunction::getSample(int i) const {
    bool isBelow = (i < 0);
    bool isAbove = (i > tableSize - 1);
    if (isBelow && (extrapolationMode == ExtrapolationMode::Linear)) {
        double slope = table[1].load() - table[0].load();
        return table[0].load() + slope * i;
    } else if (isAbove && extrapolationMode == ExtrapolationMode::Linear) {
        double slope = table[tableSize - 1].load() - table[tableSize - 2].load();
        return table[tableSize - 1].load() + slope * (i - tableSize + 1);
    } else {
        int clampedIdx = juce::jlimit(0, tableSize - 1, i);
        return table[clampedIdx].load();
    }
}

void TransferFunction::writeToStream(juce::MemoryOutputStream& stream) const {
    for (int i = 0; i < tableSize; ++i)
        stream.writeDouble(getTableValue(i));
}

void TransferFunction::readFromStream(juce::MemoryInputStream& stream) {
    for (int i = 0; i < tableSize; ++i)
        setTableValue(i, stream.readDouble());
}

void TransferFunction::applyFunction(const std::function<double(double)>& func) {
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        double y = juce::jlimit(minSignalValue, maxSignalValue, func(x));
        setTableValue(i, y);
    }
}

void TransferFunction::processBlock(double* samples, int numSamples) {
    for (int n = 0; n < numSamples; ++n) {
        double inputSample = samples[n];
        samples[n] = applyTransferFunction(inputSample);
    }
}

void TransferFunction::normalizeByMaximum() {
    double maxVal = 0.0;
    for (const auto& v : table) {
        double absVal = std::abs(v.load());
        if (absVal > maxVal)
            maxVal = absVal;
    }
    if (maxVal > 0.0) {
        for (auto& v : table) {
            v.store(v.load() / maxVal);
        }
    }
}

} // namespace dsp_core
