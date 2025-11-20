#include "TransferFunctionOperations.h"
#include <cmath>

namespace dsp_core {
namespace Services {

void TransferFunctionOperations::invert(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    // Invert all base layer values: f(x) → -f(x)
    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, -currentValue);
    }

    // Update composite and trigger visualizer update
    ltf.updateComposite();
}

void TransferFunctionOperations::removeDCInstantaneous(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    // Find DC offset at x=0 (middle of table)
    const int midIndex = tableSize / 2;
    const double dcOffset = ltf.getBaseLayerValue(midIndex);

    // Subtract DC offset from all base layer values
    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - dcOffset);
    }

    // Update composite and trigger visualizer update
    ltf.updateComposite();
}

void TransferFunctionOperations::removeDCSteadyState(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    // Calculate average value of base layer
    double sum = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        sum += ltf.getBaseLayerValue(i);
    }
    const double average = sum / tableSize;

    // Subtract average from all base layer values
    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - average);
    }

    // Update composite and trigger visualizer update
    ltf.updateComposite();
}

void TransferFunctionOperations::normalize(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    // Find maximum absolute value in base layer
    double maxAbsValue = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        double absValue = std::abs(ltf.getBaseLayerValue(i));
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }

    // Avoid division by zero
    if (maxAbsValue < 1e-10) {
        return; // Base layer is essentially zero, nothing to normalize
    }

    // Scale all base layer values so max(|f(x)|) = 1.0
    const double scaleFactor = 1.0 / maxAbsValue;
    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue * scaleFactor);
    }

    // Update composite and trigger visualizer update
    ltf.updateComposite();
}

} // namespace Services
} // namespace dsp_core
