#include "ZeroCrossingSolver.h"
#include <cmath>
#include <algorithm>

namespace dsp_core::Services {

ZeroCrossingSolver::SolveResult ZeroCrossingSolver::solve(const LayeredTransferFunction& ltf, double epsilon) {
    const int tableSize = ltf.getTableSize();

    double minAbsOutput = std::numeric_limits<double>::max();
    double bestInputValue = 0.0;
    double bestOutputValue = 0.0;

    // Brute-force search over entire table
    for (int i = 0; i < tableSize; ++i) {
        double x = ltf.normalizeIndex(i);
        double fx = ltf.applyTransferFunction(x);
        double absFx = std::abs(fx);

        if (absFx < minAbsOutput) {
            minAbsOutput = absFx;
            bestInputValue = x;
            bestOutputValue = fx;

            // Early exit if we found exact zero
            if (absFx < epsilon) {
                return {true, bestInputValue, bestOutputValue};
            }
        }
    }

    // If we found a value close to zero, refine with bisection
    if (minAbsOutput < kRefinementThreshold) {
        auto refinedResult = refineBisection(ltf, bestInputValue, kSearchRadius, epsilon, 10);

        return refinedResult;
    }

    return {false, bestInputValue, bestOutputValue};
}

ZeroCrossingSolver::SolveResult ZeroCrossingSolver::refineBisection(const LayeredTransferFunction& ltf,
                                                                    double initialGuess, double searchRadius,
                                                                    double epsilon, int maxIterations) {
    const double minSignal = ltf.getMinSignalValue();
    const double maxSignal = ltf.getMaxSignalValue();

    // Compute search bounds, respecting signal limits
    double left = std::max(minSignal, initialGuess - searchRadius);
    double right = std::min(maxSignal, initialGuess + searchRadius);

    double fLeft = ltf.applyTransferFunction(left);
    double fRight = ltf.applyTransferFunction(right);

    // Track best result during bisection
    double bestX = initialGuess;
    double bestFx = ltf.applyTransferFunction(initialGuess);
    double bestAbsFx = std::abs(bestFx);

    // Bisection iterations
    for (int iter = 0; iter < maxIterations; ++iter) {
        double mid = (left + right) * 0.5;
        double fMid = ltf.applyTransferFunction(mid);
        double absFMid = std::abs(fMid);

        // Update best if this is closer to zero
        if (absFMid < bestAbsFx) {
            bestX = mid;
            bestFx = fMid;
            bestAbsFx = absFMid;

            // Early exit if we found exact zero
            if (absFMid < epsilon) {
                return {true, bestX, bestFx};
            }
        }

        // Bisection step: choose side with opposite sign (or smaller absolute value)
        if (std::signbit(fMid) != std::signbit(fLeft)) {
            right = mid;
            fRight = fMid;
        } else {
            left = mid;
            fLeft = fMid;
        }

        // Convergence check
        if (std::abs(right - left) < epsilon) {
            break;
        }
    }

    return {bestAbsFx < epsilon, bestX, bestFx};
}

} // namespace dsp_core::Services
