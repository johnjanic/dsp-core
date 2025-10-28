#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

TEST(ZeroCrossingSolverTest, LinearOffsetCurve) {
    // f(x) = x + 0.3
    // Expected: bias ≈ -0.3
    dsp_core::LayeredTransferFunction ltf(256, -1.0, 1.0);
    for (int i = 0; i < 256; ++i) {
        double x = ltf.normalizeIndex(i);
        ltf.setBaseLayerValue(i, x + 0.3);
    }
    ltf.updateComposite();

    auto result = dsp_core::Services::ZeroCrossingSolver::solve(ltf, 0.01);

    // For linear functions, we should find a very close zero
    EXPECT_NEAR(result.inputValue, -0.3, 0.01);
    EXPECT_LT(std::abs(result.outputValue), 0.01);
}

TEST(ZeroCrossingSolverTest, NoZeroCrossing) {
    // f(x) = 0.5*tanh(x) + 0.6 (always positive)
    // Expected: finds minimum at x ≈ -1.0, residual DC > 0.05
    dsp_core::LayeredTransferFunction ltf(256, -1.0, 1.0);
    for (int i = 0; i < 256; ++i) {
        double x = ltf.normalizeIndex(i);
        ltf.setBaseLayerValue(i, 0.5 * std::tanh(x) + 0.6);
    }
    ltf.updateComposite();

    auto result = dsp_core::Services::ZeroCrossingSolver::solve(ltf);

    EXPECT_FALSE(result.hasExactZero);
    EXPECT_LT(result.inputValue, -0.9);  // Near left boundary
    EXPECT_GT(std::abs(result.outputValue), 0.05);  // Residual DC remains
}

TEST(ZeroCrossingSolverTest, MultipleZeros) {
    // f(x) = sin(3πx)
    // Expected: finds ONE zero, not necessarily the global minimum
    dsp_core::LayeredTransferFunction ltf(256, -1.0, 1.0);
    for (int i = 0; i < 256; ++i) {
        double x = ltf.normalizeIndex(i);
        ltf.setBaseLayerValue(i, std::sin(3.0 * M_PI * x));
    }
    ltf.updateComposite();

    auto result = dsp_core::Services::ZeroCrossingSolver::solve(ltf, 0.01);

    // Should find a zero crossing (output very close to zero)
    EXPECT_LT(std::abs(result.outputValue), 0.01);
}

TEST(ZeroCrossingSolverTest, TanhCurve) {
    // f(x) = tanh(3x)
    // Expected: bias ≈ 0.0 (symmetric curve)
    dsp_core::LayeredTransferFunction ltf(256, -1.0, 1.0);
    for (int i = 0; i < 256; ++i) {
        double x = ltf.normalizeIndex(i);
        ltf.setBaseLayerValue(i, std::tanh(3.0 * x));
    }
    ltf.updateComposite();

    auto result = dsp_core::Services::ZeroCrossingSolver::solve(ltf, 0.02);

    // Symmetric curve should have zero crossing near x=0
    EXPECT_NEAR(result.inputValue, 0.0, 0.02);
    EXPECT_LT(std::abs(result.outputValue), 0.02);
}
