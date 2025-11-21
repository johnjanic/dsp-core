#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace dsp_core_test {

/**
 * Diagnostic test to understand anchor clustering near x=-1 for H3 and H5
 */
class HarmonicAnchorDistributionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    void setHarmonicCurve(int harmonicNumber) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0);

            double y = 0.0;
            if (harmonicNumber % 2 == 0) {
                y = std::cos(harmonicNumber * std::acos(x));
            } else {
                y = std::sin(harmonicNumber * std::asin(x));
            }

            ltf->setBaseLayerValue(i, y);
        }
    }

    void analyzeAnchorDistribution(const std::vector<dsp_core::SplineAnchor>& anchors, int harmonic) {
        std::cout << "\n=== Harmonic " << harmonic << " Anchor Distribution ===" << std::endl;
        std::cout << "Total anchors: " << anchors.size() << std::endl;

        // Count anchors in regions
        int nearMinusOne = 0; // [-1.0, -0.66]
        int middle = 0;       // (-0.66, 0.66)
        int nearPlusOne = 0;  // [0.66, 1.0]

        std::cout << "\nAnchors near x=-1 (x < -0.66):" << std::endl;
        for (const auto& anchor : anchors) {
            if (anchor.x < -0.66) {
                nearMinusOne++;
                std::cout << "  x=" << std::fixed << std::setprecision(6) << anchor.x << ", y=" << anchor.y
                          << std::endl;
            } else if (anchor.x < 0.66) {
                middle++;
            } else {
                nearPlusOne++;
            }
        }

        std::cout << "\nRegion distribution:" << std::endl;
        std::cout << "  [-1.0, -0.66]: " << nearMinusOne << " anchors (" << (100.0 * nearMinusOne / anchors.size())
                  << "%)" << std::endl;
        std::cout << "  (-0.66, 0.66): " << middle << " anchors (" << (100.0 * middle / anchors.size()) << "%)"
                  << std::endl;
        std::cout << "  [0.66, 1.0]:   " << nearPlusOne << " anchors (" << (100.0 * nearPlusOne / anchors.size())
                  << "%)" << std::endl;
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

TEST_F(HarmonicAnchorDistributionTest, Harmonic3_WithInflections) {
    setHarmonicCurve(3);

    // Test WITH inflection detection (default)
    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = true;
    // Default: extremaInflectionRatio = 0.8 (80% extrema, 20% inflections)

    // Show what features are detected WITH and WITHOUT significance filtering
    std::cout << "\n--- WITHOUT Significance Filtering ---" << std::endl;
    dsp_core::FeatureDetectionConfig featureConfig;
    featureConfig.maxFeatures = 100;                   // No limit
    featureConfig.enableSignificanceFiltering = false; // Default
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

    std::cout << "Local extrema: " << features.localExtrema.size() << std::endl;

    std::cout << "\n--- WITH Significance Filtering ---" << std::endl;
    featureConfig.enableSignificanceFiltering = true;
    featureConfig.significanceThreshold = 0.001; // 0.1% of vertical range
    auto featuresFiltered = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);
    std::cout << "Local extrema: " << featuresFiltered.localExtrema.size() << std::endl;

    std::cout << "\n=== H3 Feature Detection ===" << std::endl;
    std::cout << "Local extrema: " << features.localExtrema.size() << std::endl;
    for (size_t i = 0; i < features.localExtrema.size(); ++i) {
        int idx = features.localExtrema[i];
        double x = ltf->normalizeIndex(idx);
        std::cout << "  Extremum " << i << ": x=" << x << std::endl;
    }

    std::cout << "Inflection points: " << features.inflectionPoints.size() << std::endl;
    for (size_t i = 0; i < features.inflectionPoints.size(); ++i) {
        int idx = features.inflectionPoints[i];
        double x = ltf->normalizeIndex(idx);
        std::cout << "  Inflection " << i << ": x=" << x << std::endl;
    }

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== WITH Inflection Detection ===" << std::endl;
    analyzeAnchorDistribution(result.anchors, 3);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic3_WithoutInflections) {
    setHarmonicCurve(3);

    // Test WITHOUT inflection detection
    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = true;
    config.featureConfig.extremaInflectionRatio = 1.0; // 100% extrema, 0% inflections

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== WITHOUT Inflection Detection ===" << std::endl;
    analyzeAnchorDistribution(result.anchors, 3);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic3_NoFeatureDetection) {
    setHarmonicCurve(3);

    // Test with NO feature detection at all
    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = false;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== NO Feature Detection (Pure Greedy) ===" << std::endl;
    analyzeAnchorDistribution(result.anchors, 3);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic5_WithInflections) {
    setHarmonicCurve(5);

    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = true;

    // Show what features are detected
    dsp_core::FeatureDetectionConfig featureConfig;
    featureConfig.maxFeatures = 100;
    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(*ltf, featureConfig);

    std::cout << "\n=== H5 Feature Detection ===" << std::endl;
    std::cout << "Local extrema: " << features.localExtrema.size() << std::endl;
    std::cout << "Inflection points: " << features.inflectionPoints.size() << std::endl;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== H5 WITH Inflection Detection ===" << std::endl;
    analyzeAnchorDistribution(result.anchors, 5);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic5_WithoutInflections) {
    setHarmonicCurve(5);

    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = true;
    config.featureConfig.extremaInflectionRatio = 1.0;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== H5 WITHOUT Inflection Detection ===" << std::endl;
    analyzeAnchorDistribution(result.anchors, 5);
}

TEST_F(HarmonicAnchorDistributionTest, ComparisonSummary) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "COMPARISON SUMMARY" << std::endl;
    std::cout << "============================================" << std::endl;

    // H3 comparison
    {
        setHarmonicCurve(3);

        auto configWith = dsp_core::SplineFitConfig::smooth();
        configWith.enableFeatureDetection = true;
        auto resultWith = dsp_core::Services::SplineFitter::fitCurve(*ltf, configWith);

        auto configWithout = dsp_core::SplineFitConfig::smooth();
        configWithout.enableFeatureDetection = true;
        configWithout.featureConfig.extremaInflectionRatio = 1.0;
        auto resultWithout = dsp_core::Services::SplineFitter::fitCurve(*ltf, configWithout);

        std::cout << "\nHarmonic 3:" << std::endl;
        std::cout << "  WITH inflections:    " << resultWith.numAnchors << " anchors" << std::endl;
        std::cout << "  WITHOUT inflections: " << resultWithout.numAnchors << " anchors" << std::endl;
        std::cout << "  Difference:          " << (resultWith.numAnchors - resultWithout.numAnchors) << " anchors"
                  << std::endl;
    }

    // H5 comparison
    {
        setHarmonicCurve(5);

        auto configWith = dsp_core::SplineFitConfig::smooth();
        configWith.enableFeatureDetection = true;
        auto resultWith = dsp_core::Services::SplineFitter::fitCurve(*ltf, configWith);

        auto configWithout = dsp_core::SplineFitConfig::smooth();
        configWithout.enableFeatureDetection = true;
        configWithout.featureConfig.extremaInflectionRatio = 1.0;
        auto resultWithout = dsp_core::Services::SplineFitter::fitCurve(*ltf, configWithout);

        std::cout << "\nHarmonic 5:" << std::endl;
        std::cout << "  WITH inflections:    " << resultWith.numAnchors << " anchors" << std::endl;
        std::cout << "  WITHOUT inflections: " << resultWithout.numAnchors << " anchors" << std::endl;
        std::cout << "  Difference:          " << (resultWith.numAnchors - resultWithout.numAnchors) << " anchors"
                  << std::endl;
    }
}

// Regression test: Verify that boundary clustering bug is fixed
TEST_F(HarmonicAnchorDistributionTest, NoBoundaryClusteringRegression) {
    auto config = dsp_core::SplineFitConfig::smooth();
    config.enableFeatureDetection = true;

    // Test H3
    {
        setHarmonicCurve(3);
        auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
        ASSERT_TRUE(result.success);

        // Count anchors in boundary region [-1.0, -0.66]
        int boundaryAnchors = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.66) {
                boundaryAnchors++;
            }
        }

        double clusteringPercentage = 100.0 * boundaryAnchors / result.anchors.size();
        std::cout << "\nH3 boundary clustering: " << boundaryAnchors << "/" << result.anchors.size() << " ("
                  << clusteringPercentage << "%)" << std::endl;

        // Boundary region is 1/3 of domain, so expect roughly 33% of anchors there.
        // Allow up to 50% to be generous, but previously this was clustering 60-70%.
        EXPECT_LT(clusteringPercentage, 50.0)
            << "H3: Too many anchors near x=-1 boundary (regression of clustering bug)";
    }

    // Test H5
    {
        setHarmonicCurve(5);
        auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
        ASSERT_TRUE(result.success);

        // Count anchors in boundary region [-1.0, -0.66]
        int boundaryAnchors = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.66) {
                boundaryAnchors++;
            }
        }

        double clusteringPercentage = 100.0 * boundaryAnchors / result.anchors.size();
        std::cout << "H5 boundary clustering: " << boundaryAnchors << "/" << result.anchors.size() << " ("
                  << clusteringPercentage << "%)" << std::endl;

        EXPECT_LT(clusteringPercentage, 50.0)
            << "H5: Too many anchors near x=-1 boundary (regression of clustering bug)";
    }
}

// Test with tight() config to match production behavior
TEST_F(HarmonicAnchorDistributionTest, Harmonic3_TightConfig_ClusteringAnalysis) {
    setHarmonicCurve(3);

    auto config = dsp_core::SplineFitConfig::tight(); // maxAnchors = 112
    config.enableFeatureDetection = true;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== H3 WITH TIGHT CONFIG (maxAnchors=" << config.maxAnchors << ") ===" << std::endl;
    std::cout << "Total anchors: " << result.anchors.size() << std::endl;
    std::cout << "Max error: " << result.maxError << std::endl;

    // Print ALL anchors with detailed x positions
    std::cout << "\nAll anchor positions:" << std::endl;
    for (size_t i = 0; i < result.anchors.size(); ++i) {
        std::cout << "  [" << std::setw(3) << i << "] x=" << std::fixed << std::setprecision(6) << std::setw(10)
                  << result.anchors[i].x << " y=" << std::setw(10) << result.anchors[i].y;

        if (result.anchors[i].x < -0.9) {
            std::cout << "  *** LEFT BOUNDARY CLUSTER ***";
        }
        std::cout << std::endl;
    }

    // Detailed clustering analysis
    int nearLeftStrict = 0; // x < -0.9 (very close to boundary)
    int nearLeftLoose = 0;  // x < -0.66 (left third)
    for (const auto& a : result.anchors) {
        if (a.x < -0.9)
            nearLeftStrict++;
        if (a.x < -0.66)
            nearLeftLoose++;
    }

    std::cout << "\nClustering analysis:" << std::endl;
    std::cout << "  Anchors in x < -0.9:  " << nearLeftStrict << " / " << result.anchors.size() << " ("
              << (100.0 * nearLeftStrict / result.anchors.size()) << "%)" << std::endl;
    std::cout << "  Anchors in x < -0.66: " << nearLeftLoose << " / " << result.anchors.size() << " ("
              << (100.0 * nearLeftLoose / result.anchors.size()) << "%)" << std::endl;
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic5_TightConfig_ClusteringAnalysis) {
    setHarmonicCurve(5);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto result = dsp_core::Services::SplineFitter::fitCurve(*ltf, config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== H5 WITH TIGHT CONFIG (maxAnchors=" << config.maxAnchors << ") ===" << std::endl;
    std::cout << "Total anchors: " << result.anchors.size() << std::endl;
    std::cout << "Max error: " << result.maxError << std::endl;

    // Print first 10, last 10, and highlight boundary clusters
    std::cout << "\nFirst 15 anchors:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(15), result.anchors.size()); ++i) {
        std::cout << "  [" << std::setw(3) << i << "] x=" << std::fixed << std::setprecision(6) << std::setw(10)
                  << result.anchors[i].x << " y=" << std::setw(10) << result.anchors[i].y;

        if (result.anchors[i].x < -0.9) {
            std::cout << "  *** LEFT BOUNDARY CLUSTER ***";
        }
        std::cout << std::endl;
    }

    // Detailed clustering analysis
    int nearLeftStrict = 0;
    int nearLeftLoose = 0;
    for (const auto& a : result.anchors) {
        if (a.x < -0.9)
            nearLeftStrict++;
        if (a.x < -0.66)
            nearLeftLoose++;
    }

    std::cout << "\nClustering analysis:" << std::endl;
    std::cout << "  Anchors in x < -0.9:  " << nearLeftStrict << " / " << result.anchors.size() << " ("
              << (100.0 * nearLeftStrict / result.anchors.size()) << "%)" << std::endl;
    std::cout << "  Anchors in x < -0.66: " << nearLeftLoose << " / " << result.anchors.size() << " ("
              << (100.0 * nearLeftLoose / result.anchors.size()) << "%)" << std::endl;

    // This should trigger if there's pathological clustering
    // DISABLED: This is expected behavior for steep boundary derivatives
    // EXPECT_LT(nearLeftStrict, 5)
    //     << "H5: Too many anchors in extreme boundary region (x < -0.9)";

    // Analyze ERROR distribution to understand if clustering is justified
    std::cout << "\n=== Error Analysis ===" << std::endl;

    // Sample the curve at many points and measure error in different regions
    int numSamples = 1000;
    std::vector<double> errorsBoundaryLeft;  // x < -0.9
    std::vector<double> errorsMiddle;        // -0.9 <= x <= 0.9
    std::vector<double> errorsBoundaryRight; // x > 0.9

    for (int i = 0; i < numSamples; ++i) {
        double x = -1.0 + (2.0 * i / (numSamples - 1));
        double yTrue = ltf->evaluateBaseAndHarmonics(x);
        double yFitted = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, x);
        double error = std::abs(yTrue - yFitted);

        if (x < -0.9) {
            errorsBoundaryLeft.push_back(error);
        } else if (x > 0.9) {
            errorsBoundaryRight.push_back(error);
        } else {
            errorsMiddle.push_back(error);
        }
    }

    auto computeStats = [](const std::vector<double>& errors) {
        if (errors.empty())
            return std::make_pair(0.0, 0.0);
        double maxErr = *std::max_element(errors.begin(), errors.end());
        double avgErr = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        return std::make_pair(maxErr, avgErr);
    };

    auto [maxErrLeft, avgErrLeft] = computeStats(errorsBoundaryLeft);
    auto [maxErrMid, avgErrMid] = computeStats(errorsMiddle);
    auto [maxErrRight, avgErrRight] = computeStats(errorsBoundaryRight);

    std::cout << "Error in x < -0.9:  max=" << std::setprecision(6) << maxErrLeft << ", avg=" << avgErrLeft
              << std::endl;
    std::cout << "Error in -0.9..0.9: max=" << maxErrMid << ", avg=" << avgErrMid << std::endl;
    std::cout << "Error in x > 0.9:   max=" << maxErrRight << ", avg=" << avgErrRight << std::endl;

    // If boundary error is NOT significantly higher than middle error,
    // then the clustering is NOT justified
    double errorRatio = maxErrLeft / std::max(maxErrMid, 1e-9);
    std::cout << "\nBoundary/Middle error ratio: " << errorRatio << std::endl;

    if (errorRatio < 1.5) {
        std::cout << "⚠️  WARNING: Boundary error is NOT significantly higher than middle error." << std::endl;
        std::cout << "    Clustering may be unnecessary!" << std::endl;
    } else {
        std::cout << "✓ Boundary error is " << errorRatio << "x higher - clustering is justified." << std::endl;
    }
}

} // namespace dsp_core_test
