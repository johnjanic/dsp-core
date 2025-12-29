#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace dsp_core_test {

/**
 * Debug test to understand why feature detector finds 0 extrema for H3/H5
 */
TEST(FeatureDetectorDebug, WhyNoExtremaForH3) {
    auto ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Set H3 curve
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        x = std::clamp(x, -1.0, 1.0);
        double y = std::sin(3.0 * std::asin(x));
        ltf->setBaseLayerValue(i, y);
    }

    std::cout << "\n=== H3 Curve Sampling ===" << std::endl;

    // Check curve near expected extremum at x ≈ 0.866
    int targetIdx = static_cast<int>(ltf->getTableSize() * (0.866 + 1.0) / 2.0);
    std::cout << "Expected extremum near x=0.866" << std::endl;
    std::cout << "Checking indices around " << targetIdx << std::endl;

    for (int offset = -5; offset <= 5; ++offset) {
        int idx = targetIdx + offset;
        if (idx >= 0 && idx < ltf->getTableSize()) {
            double x = ltf->normalizeIndex(idx);
            double y = ltf->getBaseLayerValue(idx);
            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y
                      << std::endl;
        }
    }

    // Check vertical range
    double minY = ltf->getBaseLayerValue(0);
    double maxY = ltf->getBaseLayerValue(0);
    for (int i = 1; i < ltf->getTableSize(); ++i) {
        double y = ltf->getBaseLayerValue(i);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    double verticalRange = maxY - minY;

    std::cout << "\nVertical range: [" << minY << ", " << maxY << "], span=" << verticalRange << std::endl;

    // Now run feature detection with different thresholds
    std::cout << "\n=== Feature Detection with Default Thresholds ===" << std::endl;
    {
        dsp_core::FeatureDetectionConfig config;
        std::cout << "Config: derivativeThreshold=" << config.derivativeThreshold
                  << ", significanceThreshold=" << config.significanceThreshold << std::endl;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, config);
        std::cout << "Result: " << features.localExtrema.size() << " extrema" << std::endl;
    }

    // Try with NO thresholds
    std::cout << "\n=== Feature Detection with ZERO Thresholds ===" << std::endl;
    {
        dsp_core::FeatureDetectionConfig config;
        config.derivativeThreshold = 0.0;
        config.significanceThreshold = 0.0;

        std::cout << "Config: All thresholds = 0.0" << std::endl;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, config);
        std::cout << "Result: " << features.localExtrema.size() << " extrema" << std::endl;

        for (size_t i = 0; i < std::min(size_t(10), features.localExtrema.size()); ++i) {
            int idx = features.localExtrema[i];
            double x = ltf->normalizeIndex(idx);
            double y = ltf->getBaseLayerValue(idx);
            std::cout << "  Extremum " << i << ": idx=" << idx << ", x=" << x << ", y=" << y << std::endl;
        }
    }

    // Manually check derivatives near expected extremum
    std::cout << "\n=== Manual Derivative Check at x=0.866 ===" << std::endl;
    for (int offset = -2; offset <= 2; ++offset) {
        int idx = targetIdx + offset;
        if (idx > 0 && idx < ltf->getTableSize() - 1) {
            double x = ltf->normalizeIndex(idx);
            double y = ltf->getBaseLayerValue(idx);

            // Manual central difference
            double x0 = ltf->normalizeIndex(idx - 1);
            double x1 = ltf->normalizeIndex(idx + 1);
            double y0 = ltf->getBaseLayerValue(idx - 1);
            double y1 = ltf->getBaseLayerValue(idx + 1);
            double deriv = (y1 - y0) / (x1 - x0);

            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y
                      << ", dy/dx=" << std::setprecision(8) << deriv << std::endl;
        }
    }

    // Check near the other extremum at x ≈ -0.866
    int targetIdx2 = static_cast<int>(ltf->getTableSize() * (-0.866 + 1.0) / 2.0);
    std::cout << "\n=== Manual Derivative Check at x=-0.866 ===" << std::endl;
    for (int offset = -2; offset <= 2; ++offset) {
        int idx = targetIdx2 + offset;
        if (idx > 0 && idx < ltf->getTableSize() - 1) {
            double x = ltf->normalizeIndex(idx);
            double y = ltf->getBaseLayerValue(idx);

            double x0 = ltf->normalizeIndex(idx - 1);
            double x1 = ltf->normalizeIndex(idx + 1);
            double y0 = ltf->getBaseLayerValue(idx - 1);
            double y1 = ltf->getBaseLayerValue(idx + 1);
            double deriv = (y1 - y0) / (x1 - x0);

            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y
                      << ", dy/dx=" << std::setprecision(8) << deriv << std::endl;
        }
    }
}

} // namespace dsp_core_test
