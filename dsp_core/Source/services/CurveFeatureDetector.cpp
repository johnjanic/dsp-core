#include "CurveFeatureDetector.h"
#include "../model/LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace dsp_core::Services {

// NOTE: All methods use getBaseLayerValue() instead of getCompositeValue()
// because composite might have normalization issues during fitting.

CurveFeatureDetector::FeatureResult CurveFeatureDetector::detectFeatures(const LayeredTransferFunction& ltf,
                                                                         const FeatureDetectionConfig& config) {
    FeatureResult result;
    const int tableSize = ltf.getTableSize();

    // Compute vertical metrics for significance thresholds
    double minY = ltf.getBaseLayerValue(0);
    double maxY = ltf.getBaseLayerValue(0);
    for (int i = 1; i < tableSize; ++i) {
        const double y = ltf.getBaseLayerValue(i);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    const double verticalRange = maxY - minY;
    const double amplitudeThreshold = verticalRange * config.significanceThreshold;
    const double verticalCenter = (minY + maxY) / 2.0;

    // Detect features
    std::vector<Feature> features;
    detectLocalExtrema(ltf, config, amplitudeThreshold, verticalCenter, result, features);

    // Build mandatory anchors list
    prioritizeFeatures(config, tableSize, result, features);

    // Sort and deduplicate
    std::sort(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end());
    result.mandatoryAnchors.erase(std::unique(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end()),
                                  result.mandatoryAnchors.end());

    return result;
}

void CurveFeatureDetector::detectLocalExtrema(const LayeredTransferFunction& ltf, const FeatureDetectionConfig& config,
                                              double amplitudeThreshold, double verticalCenter, FeatureResult& result,
                                              std::vector<Feature>& features) {
    const int tableSize = ltf.getTableSize();

    for (int i = 1; i < tableSize - 1; ++i) {
        const double deriv_prev = estimateDerivative(ltf, i - 1);
        const double deriv = estimateDerivative(ltf, i);

        // Detect derivative sign changes (local extrema)
        const bool hasSignChange = (deriv_prev * deriv < 0.0);
        const bool atLeastOneSignificant =
            (std::abs(deriv_prev) > config.derivativeThreshold || std::abs(deriv) > config.derivativeThreshold);

        if (!hasSignChange || !atLeastOneSignificant) {
            continue;
        }

        const double y = ltf.getBaseLayerValue(i);

        if (config.enableSignificanceFiltering) {
            // Local prominence filtering - measure how much extremum stands out
            const int windowSize = std::max(5, tableSize / 1600);
            const int windowStart = std::max(1, i - windowSize);
            const int windowEnd = std::min(tableSize - 1, i + windowSize);

            double neighborMin = std::numeric_limits<double>::max();
            double neighborMax = std::numeric_limits<double>::lowest();
            for (int j = windowStart; j <= windowEnd; ++j) {
                if (j == i)
                    continue;
                const double y_j = ltf.getBaseLayerValue(j);
                neighborMin = std::min(neighborMin, y_j);
                neighborMax = std::max(neighborMax, y_j);
            }

            // Prominence based on peak vs valley
            const double prominence = (deriv_prev > 0.0) ? (y - neighborMax) : (neighborMin - y);

            if (prominence >= amplitudeThreshold) {
                result.localExtrema.push_back(i);
                features.push_back({i, prominence});
            }
        } else {
            // Accept all extrema with valid derivative sign changes
            result.localExtrema.push_back(i);
            const double significance = std::abs(y - verticalCenter);
            features.push_back({i, significance});
        }
    }
}

void CurveFeatureDetector::prioritizeFeatures(const FeatureDetectionConfig& config, int tableSize,
                                              FeatureResult& result, std::vector<Feature>& features) {
    // Always include endpoints
    result.mandatoryAnchors.push_back(0);
    result.mandatoryAnchors.push_back(tableSize - 1);

    const bool needsPrioritization =
        (config.maxFeatures > 0 && static_cast<int>(features.size()) + 2 > config.maxFeatures);

    if (!needsPrioritization) {
        // Add all extrema features
        result.mandatoryAnchors.insert(result.mandatoryAnchors.end(), result.localExtrema.begin(),
                                       result.localExtrema.end());
        return;
    }

    // Too many features - prioritize by significance
    const int maxExtrema = config.maxFeatures - 2;

    // Sort by significance (descending)
    auto bySignificance = [](const Feature& a, const Feature& b) { return a.significance > b.significance; };
    std::sort(features.begin(), features.end(), bySignificance);

    // Keep top N most significant
    const int extremaCount = std::min(maxExtrema, static_cast<int>(features.size()));
    for (int i = 0; i < extremaCount; ++i) {
        result.mandatoryAnchors.push_back(features[i].index);
    }
}

double CurveFeatureDetector::estimateDerivative(const LayeredTransferFunction& ltf, int idx) {
    const int tableSize = ltf.getTableSize();

    // Forward difference for first point
    if (idx == 0) {
        double x0 = ltf.normalizeIndex(0);
        double x1 = ltf.normalizeIndex(1);
        double y0 = ltf.getBaseLayerValue(0);
        double y1 = ltf.getBaseLayerValue(1);
        return (y1 - y0) / (x1 - x0);
    }

    // Backward difference for last point
    if (idx == tableSize - 1) {
        double x0 = ltf.normalizeIndex(tableSize - 2);
        double x1 = ltf.normalizeIndex(tableSize - 1);
        double y0 = ltf.getBaseLayerValue(tableSize - 2);
        double y1 = ltf.getBaseLayerValue(tableSize - 1);
        return (y1 - y0) / (x1 - x0);
    }

    // Central difference for interior points
    double x0 = ltf.normalizeIndex(idx - 1);
    double x1 = ltf.normalizeIndex(idx + 1);
    double y0 = ltf.getBaseLayerValue(idx - 1);
    double y1 = ltf.getBaseLayerValue(idx + 1);

    return (y1 - y0) / (x1 - x0);
}

} // namespace dsp_core::Services
