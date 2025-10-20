#include "HarmonicLayer.h"
#include <cmath>
#include <algorithm>

namespace dsp_core {

HarmonicLayer::HarmonicLayer(int numHarmonics)
    : coefficients(numHarmonics + 1, 0.0) {
    coefficients[0] = 1.0;  // Default WT mix = 1.0 (full base layer)
}

void HarmonicLayer::setCoefficient(int harmonicIndex, double value) {
    if (harmonicIndex >= 0 && harmonicIndex < static_cast<int>(coefficients.size())) {
        coefficients[harmonicIndex] = value;
    }
}

double HarmonicLayer::getCoefficient(int harmonicIndex) const {
    if (harmonicIndex >= 0 && harmonicIndex < static_cast<int>(coefficients.size())) {
        return coefficients[harmonicIndex];
    }
    return 0.0;
}

void HarmonicLayer::setAlgorithm(Algorithm algo) {
    algorithm = algo;
}

double HarmonicLayer::evaluate(double x, int tableSize) const {
    // Use precomputed tables if available
    if (algorithm == Algorithm::Trig && precomputed && lastTableSize == tableSize) {
        int idx = xToTableIndex(x, tableSize, -1.0, 1.0);
        idx = std::max(0, std::min(tableSize - 1, idx));

        double result = 0.0;
        for (int n = 1; n < static_cast<int>(coefficients.size()); ++n) {
            result += coefficients[n] * basisFunctions[n][idx];
        }
        return result;
    }

    // Fallback to on-demand evaluation
    if (algorithm == Algorithm::Trig) {
        return evaluateChebyshevTrig(x, coefficients);
    } else {
        return evaluateChebyshevPolynomial(x, coefficients);
    }
}

void HarmonicLayer::precomputeBasisFunctions(int tableSize, double minVal, double maxVal) {
    if (precomputed && lastTableSize == tableSize) {
        return;  // Already computed
    }

    const int numHarmonics = static_cast<int>(coefficients.size()) - 1;
    basisFunctions.clear();
    basisFunctions.resize(numHarmonics + 1);

    for (int n = 0; n <= numHarmonics; ++n) {
        basisFunctions[n].resize(tableSize);

        for (int i = 0; i < tableSize; ++i) {
            // Map table index to x ∈ [minVal, maxVal]
            // Uses same formula as TransferFunction::normalizeIndex()
            double x = juce::jmap(static_cast<double>(i), 0.0, static_cast<double>(tableSize - 1), minVal, maxVal);
            x = std::max(-1.0, std::min(1.0, x));  // Clamp to valid domain

            if (n == 0) {
                basisFunctions[n][i] = 0.0;  // Reserved for WT (handled in compositor)
            } else if (n % 2 == 0) {
                // Even harmonics: cos(n * acos(x))
                basisFunctions[n][i] = std::cos(n * std::acos(x));
            } else {
                // Odd harmonics: sin(n * asin(x))
                basisFunctions[n][i] = std::sin(n * std::asin(x));
            }
        }
    }

    precomputed = true;
    lastTableSize = tableSize;
}

// Static helper functions - trig-based evaluation (copied from TransferFunctionController.cpp)
double HarmonicLayer::evaluateChebyshevTrig(double x, const std::vector<double>& coeffs) {
    double result = 0.0;
    const int N = static_cast<int>(coeffs.size());
    x = std::max(-1.0, std::min(1.0, x));  // Clamp to valid domain

    for (int n = 1; n < N; ++n) {
        double term = 0.0;
        if (n % 2 == 0) {
            // Even harmonics: cos(n * acos(x))
            term = std::cos(n * std::acos(x));
        } else {
            // Odd harmonics: sin(n * asin(x))
            term = std::sin(n * std::asin(x));
        }
        result += coeffs[n] * term;
    }
    return result;
}

// Polynomial evaluation using Clenshaw's algorithm (future support)
double HarmonicLayer::evaluateChebyshevPolynomial(double x, const std::vector<double>& coeffs) {
    if (coeffs.size() <= 1) return 0.0;

    // Clenshaw's algorithm for numerical stability
    double b_kplus1 = 0.0;
    double b_kplus2 = 0.0;
    int N = static_cast<int>(coeffs.size()) - 1;

    for (int k = N; k >= 1; --k) {
        double temp = b_kplus1;
        b_kplus1 = 2.0 * x * b_kplus1 - b_kplus2 + coeffs[k];
        b_kplus2 = temp;
    }
    return x * b_kplus1 - b_kplus2;
}

juce::ValueTree HarmonicLayer::toValueTree() const {
    juce::ValueTree vt("HarmonicLayer");
    vt.setProperty("algorithm", algorithm == Algorithm::Trig ? "trig" : "polynomial", nullptr);

    juce::Array<juce::var> coeffArray;
    for (double c : coefficients) {
        coeffArray.add(c);
    }
    vt.setProperty("coefficients", coeffArray, nullptr);
    return vt;
}

void HarmonicLayer::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "HarmonicLayer") {
        return;
    }

    // Load algorithm
    juce::String algoStr = vt.getProperty("algorithm", "trig").toString();
    algorithm = (algoStr == "polynomial") ? Algorithm::Polynomial : Algorithm::Trig;

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

    // Invalidate precomputed data (will be recomputed on next evaluate)
    precomputed = false;
}

bool HarmonicLayer::operator==(const HarmonicLayer& other) const {
    return coefficients == other.coefficients && algorithm == other.algorithm;
}

int HarmonicLayer::xToTableIndex(double x, int tableSize, double minVal, double maxVal) const {
    // Map x from [minVal, maxVal] to table index [0, tableSize-1]
    // Inverse of the jmap operation used in precomputeBasisFunctions
    return static_cast<int>(juce::jmap(x, minVal, maxVal, 0.0, static_cast<double>(tableSize - 1)));
}

} // namespace dsp_core
