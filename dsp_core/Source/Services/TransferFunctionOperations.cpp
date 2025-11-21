#include "TransferFunctionOperations.h"
#include <cmath>

namespace dsp_core {
namespace Services {

void TransferFunctionOperations::invert(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, -currentValue);
    }

    ltf.updateComposite();
}

void TransferFunctionOperations::removeDCInstantaneous(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();
    const int midIndex = tableSize / 2;
    const double dcOffset = ltf.getBaseLayerValue(midIndex);

    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - dcOffset);
    }

    ltf.updateComposite();
}

void TransferFunctionOperations::removeDCSteadyState(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    double sum = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        sum += ltf.getBaseLayerValue(i);
    }
    const double average = sum / tableSize;

    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - average);
    }

    ltf.updateComposite();
}

void TransferFunctionOperations::normalize(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    double maxAbsValue = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        double absValue = std::abs(ltf.getBaseLayerValue(i));
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }

    if (maxAbsValue < 1e-10) {
        return;
    }

    const double scaleFactor = 1.0 / maxAbsValue;
    for (int i = 0; i < tableSize; ++i) {
        double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue * scaleFactor);
    }

    ltf.updateComposite();
}

} // namespace Services
} // namespace dsp_core
