#include "CurveFeatureDetector.h"
#include "../LayeredTransferFunction.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace dsp_core {
namespace Services {

CurveFeatureDetector::FeatureResult CurveFeatureDetector::detectFeatures(const LayeredTransferFunction& ltf,
                                                                        const FeatureDetectionConfig& config) {
    FeatureResult result;
    int tableSize = ltf.getTableSize();

    // Compute vertical range for significance filtering
    // Use base layer for testing (composite might have normalization issues)
    double minY = ltf.getBaseLayerValue(0);
    double maxY = ltf.getBaseLayerValue(0);
    for (int i = 1; i < tableSize; ++i) {
        double y = ltf.getBaseLayerValue(i);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    double verticalRange = maxY - minY;
    double amplitudeThreshold = verticalRange * config.significanceThreshold;
    double verticalCenter = (minY + maxY) / 2.0;

    // 1. Find local extrema (dy/dx sign changes) with significance scores
    struct Feature {
        int index;
        double significance;  // For extrema: |y|, for inflection: |d²y/dx²|
        bool isExtremum;      // true = extremum, false = inflection point
    };
    std::vector<Feature> features;

    for (int i = 1; i < tableSize - 1; ++i) {
        double deriv_prev = estimateDerivative(ltf, i - 1);
        double deriv = estimateDerivative(ltf, i);

        // Detect derivative sign changes (local extrema)
        // Note: At extrema, one derivative may be near zero, so we check if
        // at least one is significant OR if there's a clear sign change
        bool hasSignChange = (deriv_prev * deriv < 0.0);
        bool atLeastOneSignificant = (std::abs(deriv_prev) > config.derivativeThreshold ||
                                      std::abs(deriv) > config.derivativeThreshold);

        if (hasSignChange && atLeastOneSignificant) {
            // Use base layer for testing (composite might have normalization issues)
            double y = ltf.getBaseLayerValue(i);

            if (config.enableSignificanceFiltering) {
                // EXPERIMENTAL: Local prominence filtering
                // Measure how much this extremum stands out from nearby points
                // A peak at y=0 is just as significant as a peak at y=1!

                // Define adaptive window size for prominence measurement
                // For 16k samples: ~10 samples = 0.06% of domain
                // Conservative window size to preserve smooth extrema
                int windowSize = std::max(5, tableSize / 1600);
                int windowStart = std::max(1, i - windowSize);
                int windowEnd = std::min(tableSize - 1, i + windowSize);

                // Find min/max in neighborhood (excluding the extremum itself)
                // Use sentinel values to handle edge case where windowStart == i
                double neighborMin = std::numeric_limits<double>::max();
                double neighborMax = std::numeric_limits<double>::lowest();
                for (int j = windowStart; j <= windowEnd; ++j) {
                    if (j == i) continue;  // Skip the extremum itself
                    double y_j = ltf.getBaseLayerValue(j);
                    neighborMin = std::min(neighborMin, y_j);
                    neighborMax = std::max(neighborMax, y_j);
                }

                // Prominence: how much does this extremum stand out from neighbors?
                double prominence;
                if (deriv_prev > 0.0) {  // Peak (derivative changes from + to -)
                    prominence = y - neighborMax;
                } else {  // Valley (derivative changes from - to +)
                    prominence = neighborMin - y;
                }

                // Apply significance filtering based on local prominence
                if (prominence >= amplitudeThreshold) {
                    result.localExtrema.push_back(i);
                    features.push_back({i, prominence, true});
                }
            } else {
                // Default: Accept all extrema with valid derivative sign changes
                // The derivative threshold already filtered numerical noise
                result.localExtrema.push_back(i);

                // For prioritization when hitting maxFeatures limit
                double significance = std::abs(y - verticalCenter);
                features.push_back({i, significance, true});
            }
        }
    }

    // 2. Find inflection points (d²y/dx² sign changes) with curvature scores
    for (int i = 2; i < tableSize - 2; ++i) {
        double d2y_prev = estimateSecondDerivative(ltf, i - 1);
        double d2y = estimateSecondDerivative(ltf, i);

        // Adaptive threshold: stricter near boundaries to avoid false inflections
        // where derivatives explode (common for odd harmonics)
        double x = ltf.normalizeIndex(i);
        double distanceFromBoundary = std::min(std::abs(x - (-1.0)), std::abs(x - 1.0));
        double threshold = config.secondDerivativeThreshold;

        // Increase threshold by 5x within 0.1 units of boundary
        if (distanceFromBoundary < 0.1) {
            threshold *= 5.0;
        }

        // Only detect sign changes if both second derivatives are significant
        if (std::abs(d2y_prev) > threshold &&
            std::abs(d2y) > threshold &&
            d2y_prev * d2y < 0.0) {  // Sign change = inflection point
            result.inflectionPoints.push_back(i);

            // Significance = curvature magnitude
            features.push_back({i, std::abs(d2y), false});
        }
    }

    // 3. Prioritize and limit features if maxFeatures is set
    result.mandatoryAnchors.push_back(0);  // Always include endpoints
    result.mandatoryAnchors.push_back(tableSize - 1);

    if (config.maxFeatures > 0 && static_cast<int>(features.size()) + 2 > config.maxFeatures) {
        // Too many features - prioritize by significance
        // Reserve extremaInflectionRatio for extrema (peaks/valleys are typically more important than inflection points)
        int maxExtrema = static_cast<int>((config.maxFeatures - 2) * config.extremaInflectionRatio);
        int maxInflections = (config.maxFeatures - 2) - maxExtrema;

        // Separate extrema and inflections
        std::vector<Feature> extrema, inflections;
        for (const auto& f : features) {
            if (f.isExtremum) extrema.push_back(f);
            else inflections.push_back(f);
        }

        // Sort by significance (descending)
        auto bySignificance = [](const Feature& a, const Feature& b) {
            return a.significance > b.significance;
        };
        std::sort(extrema.begin(), extrema.end(), bySignificance);
        std::sort(inflections.begin(), inflections.end(), bySignificance);

        // Keep top N most significant
        for (int i = 0; i < std::min(maxExtrema, static_cast<int>(extrema.size())); ++i) {
            result.mandatoryAnchors.push_back(extrema[i].index);
        }
        for (int i = 0; i < std::min(maxInflections, static_cast<int>(inflections.size())); ++i) {
            result.mandatoryAnchors.push_back(inflections[i].index);
        }
    } else {
        // Not limited - add all features
        result.mandatoryAnchors.insert(
            result.mandatoryAnchors.end(),
            result.localExtrema.begin(),
            result.localExtrema.end()
        );
        result.mandatoryAnchors.insert(
            result.mandatoryAnchors.end(),
            result.inflectionPoints.begin(),
            result.inflectionPoints.end()
        );
    }

    // 4. Sort and deduplicate
    std::sort(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end());
    result.mandatoryAnchors.erase(
        std::unique(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end()),
        result.mandatoryAnchors.end()
    );

    return result;
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

    // Central difference for interior points (more accurate)
    double x0 = ltf.normalizeIndex(idx - 1);
    double x1 = ltf.normalizeIndex(idx + 1);
    // Use base layer for testing (composite might have normalization issues)
    double y0 = ltf.getBaseLayerValue(idx - 1);
    double y1 = ltf.getBaseLayerValue(idx + 1);

    return (y1 - y0) / (x1 - x0);
}

double CurveFeatureDetector::estimateSecondDerivative(const LayeredTransferFunction& ltf, int idx) {
    if (idx < 1 || idx >= ltf.getTableSize() - 1)
        return 0.0;

    double h = ltf.normalizeIndex(1) - ltf.normalizeIndex(0);  // Uniform spacing
    // Use base layer for testing (composite might have normalization issues)
    double y_prev = ltf.getBaseLayerValue(idx - 1);
    double y = ltf.getBaseLayerValue(idx);
    double y_next = ltf.getBaseLayerValue(idx + 1);

    return (y_next - 2.0 * y + y_prev) / (h * h);
}

// Legacy overload for backward compatibility
CurveFeatureDetector::FeatureResult CurveFeatureDetector::detectFeatures(const LayeredTransferFunction& ltf,
                                                                         int maxMandatoryAnchors) {
    FeatureDetectionConfig config;
    config.maxFeatures = maxMandatoryAnchors;
    return detectFeatures(ltf, config);
}

} // namespace Services
} // namespace dsp_core
