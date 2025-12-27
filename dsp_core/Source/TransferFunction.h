#pragma once

#include <vector>
#include <atomic>
#include <juce_core/juce_core.h>

namespace dsp_core {

class TransferFunction {
  public:
    enum class InterpolationMode { Linear, Cubic, CatmullRom };

    enum class ExtrapolationMode { Clamp, Linear };

    TransferFunction(int tableSize, double minSignalValue = -1.0, double maxSignalValue = 1.0);

    void reset();
    double normalizeIndex(int index) const;

    double applyTransferFunction(double x) const;

    void setInterpolationMode(InterpolationMode mode);
    void setExtrapolationMode(ExtrapolationMode mode);

    void setTableValue(int index, double value);
    void setTableValues(const double* samples, const int numSamples);
    double getTableValue(int index) const;
    int getTableIndex(double x) const {
        return static_cast<int>(juce::jmap(x, minSignalValue, maxSignalValue, 0.0, static_cast<double>(tableSize - 1)));
    }

    int getTableSize() const {
        return tableSize;
    }
    double getMinSignalValue() const {
        return minSignalValue;
    }
    double getMaxSignalValue() const {
        return maxSignalValue;
    }

    // For GUI/editor access
    const std::vector<std::atomic<double>>& getTable() const {
        return table;
    }
    std::vector<std::atomic<double>>& getTable() {
        return table;
    }

    void processBlock(double* samples, int numSamples) const;

  private:
    const double minExtrapolationAmmount = -10.0;
    const double maxExtrapolationAmmount = 10.0;

    double getSample(int i) const;
    double applyTransferFunctionLinear(double x) const;
    double applyTransferFunctionCubic(double x) const;
    double applyTransferFunctionCatmullRom(double x) const;
    static double interpolateLinear(double y0, double y1, double t);
    static double interpolateCubic(double y0, double y1, double y2, double y3, double t);
    static double interpolateCatmullRom(double y0, double y1, double y2, double y3, double t);

    int tableSize;
    double minSignalValue, maxSignalValue;
    std::vector<std::atomic<double>> table;
    InterpolationMode interpolationMode = InterpolationMode::Cubic;
    ExtrapolationMode extrapolationMode = ExtrapolationMode::Clamp;
};

} // namespace dsp_core
