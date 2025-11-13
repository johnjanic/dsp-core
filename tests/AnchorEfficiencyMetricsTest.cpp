#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Anchor Efficiency Metrics Test Suite
 *
 * Measures anchor counts for various curve types to evaluate spline fitting efficiency.
 * Lower anchor counts are better (more efficient representation).
 */
class AnchorEfficiencyMetrics : public ::testing::Test {
protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(4096, -1.0, 1.0);
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

//==============================================================================
// Helper: Set curve types
//==============================================================================

void setTanhCurve(dsp_core::LayeredTransferFunction& ltf, double steepness = 2.0) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double x = ltf.normalizeIndex(i);
        double y = std::tanh(steepness * x);
        ltf.setBaseLayerValue(i, y);
    }
}

void setSinCurve(dsp_core::LayeredTransferFunction& ltf, double frequency = 1.0) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double x = ltf.normalizeIndex(i);
        double y = std::sin(frequency * M_PI * x);
        ltf.setBaseLayerValue(i, y);
    }
}

void setHarmonicCurve(dsp_core::LayeredTransferFunction& ltf, int harmonicNumber) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double x = ltf.normalizeIndex(i);
        x = std::max(-1.0, std::min(1.0, x));  // Clamp

        double y = 0.0;
        if (harmonicNumber % 2 == 0) {
            // Even harmonics: cos(n * acos(x))
            y = std::cos(harmonicNumber * std::acos(x));
        } else {
            // Odd harmonics: sin(n * asin(x))
            y = std::sin(harmonicNumber * std::asin(x));
        }

        ltf.setBaseLayerValue(i, y);
    }
}

//==============================================================================
// Anchor Count Tests
//==============================================================================

TEST_F(AnchorEfficiencyMetrics, Tanh_AnchorCount) {
    setTanhCurve(*ltf, 2.0);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: Tanh_AnchorCount = " << anchorCount << std::endl;

    // Sanity check - should use reasonable number of anchors
    EXPECT_GE(anchorCount, 3) << "Too few anchors for tanh";
    EXPECT_LE(anchorCount, 20) << "Too many anchors for tanh";
}

TEST_F(AnchorEfficiencyMetrics, Sin_AnchorCount) {
    setSinCurve(*ltf, 1.0);  // sin(πx)

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: Sin_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 4) << "Too few anchors for sin";
    EXPECT_LE(anchorCount, 15) << "Too many anchors for sin";
}

TEST_F(AnchorEfficiencyMetrics, H3_AnchorCount) {
    setHarmonicCurve(*ltf, 3);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H3_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 4) << "Too few anchors for H3";
    EXPECT_LE(anchorCount, 20) << "Too many anchors for H3";
}

TEST_F(AnchorEfficiencyMetrics, H5_AnchorCount) {
    setHarmonicCurve(*ltf, 5);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H5_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 6) << "Too few anchors for H5";
    EXPECT_LE(anchorCount, 30) << "Too many anchors for H5";
}

TEST_F(AnchorEfficiencyMetrics, H10_AnchorCount) {
    setHarmonicCurve(*ltf, 10);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H10_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 10) << "Too few anchors for H10";
    EXPECT_LE(anchorCount, 50) << "Too many anchors for H10";
}

TEST_F(AnchorEfficiencyMetrics, H15_AnchorCount) {
    setHarmonicCurve(*ltf, 15);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H15_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 15) << "Too few anchors for H15";
    EXPECT_LE(anchorCount, 70) << "Too many anchors for H15";
}

TEST_F(AnchorEfficiencyMetrics, H20_AnchorCount) {
    setHarmonicCurve(*ltf, 20);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H20_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 20) << "Too few anchors for H20";
    EXPECT_LE(anchorCount, 80) << "Too many anchors for H20";
}

TEST_F(AnchorEfficiencyMetrics, H30_AnchorCount) {
    setHarmonicCurve(*ltf, 30);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H30_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 30) << "Too few anchors for H30";
    EXPECT_LE(anchorCount, 110) << "Too many anchors for H30";
}

TEST_F(AnchorEfficiencyMetrics, H40_AnchorCount) {
    setHarmonicCurve(*ltf, 40);

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success);

    int anchorCount = static_cast<int>(result.anchors.size());
    std::cout << "METRIC: H40_AnchorCount = " << anchorCount << std::endl;

    EXPECT_GE(anchorCount, 40) << "Too few anchors for H40";
    EXPECT_LE(anchorCount, 130) << "Too many anchors for H40";
}

} // namespace dsp_core_test
