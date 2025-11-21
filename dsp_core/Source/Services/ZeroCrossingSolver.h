#pragma once
#include "../LayeredTransferFunction.h"

namespace dsp_core::Services {

/**
 * Service: Zero-Crossing Solver for Transfer Functions
 *
 * Finds the input value x where f(x) is closest to zero.
 * Used by DC offset compensation to determine bias amount.
 *
 * Algorithm:
 *   1. Brute-force search over table resolution
 *   2. Optional bisection refinement if close to zero
 *   3. Returns input value (bias) and output value (residual DC)
 *
 * Thread Safety: UI thread only (not real-time safe due to function evaluations)
 */
class ZeroCrossingSolver {
  public:
    struct SolveResult {
        bool hasExactZero;  // True if |f(x)| < epsilon
        double inputValue;  // x* where f(x*) ≈ 0 (the bias)
        double outputValue; // Actual f(x*) (residual DC offset)
    };

    /**
     * Find the input value where f(x) is closest to zero.
     *
     * @param ltf The transfer function to analyze
     * @param epsilon Threshold for "exact zero" detection (default: 1e-9)
     * @return SolveResult containing bias and residual DC
     */
    static SolveResult solve(const LayeredTransferFunction& ltf, double epsilon = 1e-9);

  private:
    ZeroCrossingSolver() = delete; // Pure static service

    /**
     * Refine zero-crossing estimate using bisection.
     * Only called if brute-force finds |f(x)| < refinementThreshold.
     */
    static SolveResult refineBisection(const LayeredTransferFunction& ltf, double initialGuess, double searchRadius,
                                       double epsilon, int maxIterations = 10);

    static constexpr double kRefinementThreshold = 0.01;
    static constexpr double kSearchRadius = 0.01;
};

} // namespace dsp_core::Services
