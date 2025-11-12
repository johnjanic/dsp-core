#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

namespace dsp_core_test {

/**
 * Diagnostic test to understand 4→6 backtranslation regression
 *
 * This test reproduces the ArbitraryPositions scenario and reports
 * exactly what features are detected and where anchors are placed.
 */
TEST(FeatureDetectionDiagnostic, ArbitraryPositions_FeatureAnalysis) {
    std::cout << "\n=== Feature Detection Diagnostic: ArbitraryPositions ===" << std::endl;

    // Step 1: Create the exact same 4-anchor curve from ArbitraryPositions test
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, -0.87, false, 0.0},
        {-0.412, -0.653, false, 0.0},
        {0.234, 0.123, false, 0.0},
        {1.0, 0.92, false, 0.0}
    };

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    std::cout << "\nOriginal anchors (4):" << std::endl;
    for (size_t i = 0; i < originalAnchors.size(); ++i) {
        std::cout << "  [" << i << "] x=" << std::setw(8) << std::fixed << std::setprecision(4)
                  << originalAnchors[i].x << ", y=" << std::setw(8) << originalAnchors[i].y
                  << ", m=" << std::setw(8) << originalAnchors[i].tangent << std::endl;
    }

    // Step 2: Render to LayeredTransferFunction (like backtranslation does)
    auto ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Render spline to base layer
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        double y = dsp_core::Services::SplineEvaluator::evaluate(originalAnchors, x);
        ltf->setBaseLayerValue(i, y);
    }

    std::cout << "\nRendered to LTF with " << ltf->getTableSize() << " samples" << std::endl;

    // Step 3: Run feature detection with various configurations
    std::cout << "\n--- Feature Detection Analysis ---" << std::endl;

    // Test A: Zero threshold (detect everything)
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.0;  // Detect ALL features
        featureConfig.derivativeThreshold = 1e-6;
        featureConfig.secondDerivativeThreshold = 1e-4;
        featureConfig.extremaInflectionRatio = 0.8;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

        std::cout << "\nConfig: significanceThreshold=0.0 (detect all)" << std::endl;
        std::cout << "  Local extrema: " << features.localExtrema.size() << std::endl;
        for (size_t i = 0; i < features.localExtrema.size(); ++i) {
            int idx = features.localExtrema[i];
            double x = ltf->normalizeIndex(idx);
            double y = ltf->evaluateBaseAndHarmonics(x);
            std::cout << "    [" << i << "] idx=" << idx << ", x=" << std::setprecision(6) << x
                      << ", y=" << y << std::endl;
        }

        std::cout << "  Inflection points: " << features.inflectionPoints.size() << std::endl;
        for (size_t i = 0; i < features.inflectionPoints.size(); ++i) {
            int idx = features.inflectionPoints[i];
            double x = ltf->normalizeIndex(idx);
            double y = ltf->evaluateBaseAndHarmonics(x);
            std::cout << "    [" << i << "] idx=" << idx << ", x=" << std::setprecision(6) << x
                      << ", y=" << y << std::endl;
        }

        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << std::endl;
    }

    // Test B: Production threshold (0.001)
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.001;
        featureConfig.derivativeThreshold = 1e-6;
        featureConfig.secondDerivativeThreshold = 1e-4;
        featureConfig.extremaInflectionRatio = 0.8;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

        std::cout << "\nConfig: significanceThreshold=0.001 (production)" << std::endl;
        std::cout << "  Local extrema: " << features.localExtrema.size() << std::endl;
        std::cout << "  Inflection points: " << features.inflectionPoints.size() << std::endl;
        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << std::endl;
    }

    // Test C: Higher derivative threshold
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.001;
        featureConfig.derivativeThreshold = 1e-5;  // 10x higher
        featureConfig.secondDerivativeThreshold = 1e-3;  // 10x higher
        featureConfig.extremaInflectionRatio = 0.8;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

        std::cout << "\nConfig: Higher derivative thresholds (1e-5, 1e-3)" << std::endl;
        std::cout << "  Local extrema: " << features.localExtrema.size() << std::endl;
        std::cout << "  Inflection points: " << features.inflectionPoints.size() << std::endl;
        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << std::endl;
    }

    // Test D: Extrema only (no inflections)
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.001;
        featureConfig.derivativeThreshold = 1e-6;
        featureConfig.secondDerivativeThreshold = 1e-3;  // Very high → ignore inflections
        featureConfig.extremaInflectionRatio = 1.0;  // 100% extrema, 0% inflections

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

        std::cout << "\nConfig: Extrema only (extremaInflectionRatio=1.0)" << std::endl;
        std::cout << "  Local extrema: " << features.localExtrema.size() << std::endl;
        std::cout << "  Inflection points: " << features.inflectionPoints.size() << std::endl;
        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << std::endl;
    }

    // Step 4: Refit with feature detection enabled
    std::cout << "\n--- Spline Refitting ---" << std::endl;

    {
        auto fitConfig = dsp_core::SplineFitConfig::tight();
        fitConfig.enableFeatureDetection = true;
        fitConfig.featureConfig.significanceThreshold = 0.001;
        fitConfig.featureConfig.derivativeThreshold = 1e-6;
        fitConfig.featureConfig.secondDerivativeThreshold = 1e-4;

        auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, fitConfig);

        std::cout << "\nRefit with feature detection: " << result.anchors.size() << " anchors" << std::endl;
        for (size_t i = 0; i < result.anchors.size(); ++i) {
            std::cout << "  [" << i << "] x=" << std::setw(8) << std::fixed << std::setprecision(4)
                      << result.anchors[i].x << ", y=" << std::setw(8) << result.anchors[i].y << std::endl;
        }
    }

    // Step 5: Refit with extrema-only feature detection
    {
        auto fitConfig = dsp_core::SplineFitConfig::tight();
        fitConfig.enableFeatureDetection = true;
        fitConfig.featureConfig.significanceThreshold = 0.001;
        fitConfig.featureConfig.derivativeThreshold = 1e-6;
        fitConfig.featureConfig.secondDerivativeThreshold = 1e-3;  // Ignore inflections
        fitConfig.featureConfig.extremaInflectionRatio = 1.0;

        auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, fitConfig);

        std::cout << "\nRefit with EXTREMA ONLY: " << result.anchors.size() << " anchors" << std::endl;
        for (size_t i = 0; i < result.anchors.size(); ++i) {
            std::cout << "  [" << i << "] x=" << std::setw(8) << std::fixed << std::setprecision(4)
                      << result.anchors[i].x << ", y=" << std::setw(8) << result.anchors[i].y << std::endl;
        }
    }

    std::cout << "\n=== Analysis Complete ===" << std::endl;
}

} // namespace dsp_core_test
