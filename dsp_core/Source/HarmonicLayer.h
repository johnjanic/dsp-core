#pragma once
#include <juce_core/juce_core.h>
#include <vector>

namespace dsp_core {

/**
 * HarmonicLayer - Harmonic waveshaping coefficients with optimized evaluation
 *
 * Stores harmonic coefficients and evaluates Chebyshev-based waveshaping functions.
 * Uses precomputed basis functions for 10-50x performance improvement.
 *
 * coefficients[0] = WT (wavetable mix) - controls base layer blending
 * coefficients[1..N] = harmonic amplitudes
 */
class HarmonicLayer {
public:
    enum class Algorithm {
        Trig,       // Default: sin(n*asin(x)) for odd, cos(n*acos(x)) for even
        Polynomial  // Alternative: Clenshaw's algorithm for Chebyshev polynomials
    };

    explicit HarmonicLayer(int numHarmonics = 19);

    //==========================================================================
    // Coefficient Access
    //==========================================================================

    void setCoefficient(int harmonicIndex, double value);
    double getCoefficient(int harmonicIndex) const;
    int getNumHarmonics() const { return static_cast<int>(coefficients.size()) - 1; }

    void setAlgorithm(Algorithm algo);
    Algorithm getAlgorithm() const { return algorithm; }

    //==========================================================================
    // Evaluation (thread-safe read)
    //==========================================================================

    /**
     * Evaluate harmonic contribution at normalized position x ∈ [-1, 1]
     *
     * @param x Normalized position in transfer function domain
     * @param tableSize Size of the transfer function table (for basis lookup)
     * @return Sum of weighted harmonics (does NOT include WT * base)
     */
    double evaluate(double x, int tableSize) const;

    //==========================================================================
    // Precomputation (call once, or when table size changes)
    //==========================================================================

    /**
     * Precompute basis functions for all harmonics across table
     *
     * Stores sin(n*asin(x)) and cos(n*acos(x)) for all x values,
     * eliminating trig calculations during slider movements.
     *
     * @param tableSize Number of points in transfer function table
     * @param minVal Minimum value of transfer function domain (typically -1.0)
     * @param maxVal Maximum value of transfer function domain (typically +1.0)
     */
    void precomputeBasisFunctions(int tableSize, double minVal, double maxVal);

    //==========================================================================
    // Serialization
    //==========================================================================

    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

    bool operator==(const HarmonicLayer& other) const;

private:
    std::vector<double> coefficients;  // [0] = WT mix, [1..N] = harmonics
    Algorithm algorithm = Algorithm::Trig;

    // Precomputed basis functions: [harmonic_index][table_index]
    mutable std::vector<std::vector<double>> basisFunctions;
    mutable bool precomputed = false;
    mutable int lastTableSize = 0;

    // Chebyshev evaluation functions
    static double evaluateChebyshevTrig(double x, const std::vector<double>& coeffs);
    static double evaluateChebyshevPolynomial(double x, const std::vector<double>& coeffs);

    int xToTableIndex(double x, int tableSize, double minVal, double maxVal) const;
};

} // namespace dsp_core
