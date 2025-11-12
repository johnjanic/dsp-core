#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

namespace dsp_core_test {

/**
 * Test fixture specifically for outputting parseable quality metrics
 *
 * This test file exists solely to provide machine-readable output for
 * the parameter tuning script. It computes and prints extrema position
 * errors in a format that can be parsed.
 */
class ExtremaQualityMetrics : public ::testing::Test {
protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;

    /**
     * Find numerical extrema by scanning dy/dx sign changes
     */
    std::vector<double> findNumericalExtrema(const dsp_core::LayeredTransferFunction& ltf) {
        std::vector<double> extrema;
        const int tableSize = ltf.getTableSize();

        for (int i = 1; i < tableSize - 1; ++i) {
            double x_prev = ltf.normalizeIndex(i - 1);
            double x_curr = ltf.normalizeIndex(i);
            double x_next = ltf.normalizeIndex(i + 1);

            double y_prev = ltf.evaluateBaseAndHarmonics(x_prev);
            double y_curr = ltf.evaluateBaseAndHarmonics(x_curr);
            double y_next = ltf.evaluateBaseAndHarmonics(x_next);

            double slope_left = (y_curr - y_prev) / (x_curr - x_prev);
            double slope_right = (y_next - y_curr) / (x_next - x_curr);

            // Extremum: derivative sign change
            if (slope_left * slope_right < 0) {
                extrema.push_back(x_curr);
            }
        }

        return extrema;
    }

    /**
     * Find anchor closest to target x position
     */
    double findAnchorNear(const std::vector<dsp_core::SplineAnchor>& anchors, double target_x) {
        if (anchors.empty()) {
            return std::numeric_limits<double>::infinity();
        }

        double closest_x = anchors[0].x;
        double min_dist = std::abs(anchors[0].x - target_x);

        for (const auto& anchor : anchors) {
            double dist = std::abs(anchor.x - target_x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_x = anchor.x;
            }
        }

        return closest_x;
    }

    /**
     * Compute average position error for a set of extrema
     */
    double computeAveragePositionError(const std::vector<dsp_core::SplineAnchor>& anchors,
                                      const std::vector<double>& extrema) {
        if (extrema.empty()) return 0.0;

        double total_error = 0.0;
        for (double extremum_x : extrema) {
            double anchor_x = findAnchorNear(anchors, extremum_x);
            double error = std::abs(anchor_x - extremum_x);
            total_error += error;
        }

        return total_error / extrema.size();
    }
};

/**
 * Harmonic 3 extrema quality metric
 *
 * Output format: "METRIC: Harmonic3_PositionError = 0.001234"
 */
TEST_F(ExtremaQualityMetrics, Harmonic3_PositionError) {
    // Create Harmonic 3: sin(3*asin(x))
    for (int i = 0; i < 16384; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = std::sin(3.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    // Find numerical extrema
    auto extrema = findNumericalExtrema(*ltf);

    // Fit with current configuration (inherits from SplineFitConfig::tight())
    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Harmonic 3 fit failed";

    // Compute average position error
    double avg_error = computeAveragePositionError(result.anchors, extrema);

    // Output in parseable format
    std::cout << "METRIC: Harmonic3_PositionError = "
              << std::fixed << std::setprecision(6) << avg_error << std::endl;

    // Still assert for test pass/fail
    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 5 extrema quality metric
 */
TEST_F(ExtremaQualityMetrics, Harmonic5_PositionError) {
    // Create Harmonic 5
    for (int i = 0; i < 16384; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = std::sin(5.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Harmonic 5 fit failed";

    double avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic5_PositionError = "
              << std::fixed << std::setprecision(6) << avg_error << std::endl;

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 10 extrema quality metric
 */
TEST_F(ExtremaQualityMetrics, Harmonic10_PositionError) {
    // Create Harmonic 10
    for (int i = 0; i < 16384; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = std::sin(10.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Harmonic 10 fit failed";

    double avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic10_PositionError = "
              << std::fixed << std::setprecision(6) << avg_error << std::endl;

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 20 extrema quality metric (HIGH FREQUENCY)
 */
TEST_F(ExtremaQualityMetrics, Harmonic20_PositionError) {
    // Create Harmonic 20
    for (int i = 0; i < 16384; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = std::sin(20.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Harmonic 20 fit failed";

    double avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic20_PositionError = "
              << std::fixed << std::setprecision(6) << avg_error << std::endl;

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 40 extrema quality metric (VERY HIGH FREQUENCY - stress test)
 */
TEST_F(ExtremaQualityMetrics, Harmonic40_PositionError) {
    // Create Harmonic 40
    for (int i = 0; i < 16384; ++i) {
        double x = ltf->normalizeIndex(i);
        double y = std::sin(40.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);

    ASSERT_TRUE(result.success) << "Harmonic 40 fit failed";

    double avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic40_PositionError = "
              << std::fixed << std::setprecision(6) << avg_error << std::endl;

    EXPECT_LT(avg_error, 0.15) << "Position error should be reasonable (relaxed for H40)";
}

} // namespace dsp_core_test
