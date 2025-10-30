#include "CurveFeatureDetector.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {
namespace Services {

CurveFeatureDetector::FeatureResult CurveFeatureDetector::detectFeatures(const LayeredTransferFunction& ltf, int maxMandatoryAnchors) {
    FeatureResult result;
    int tableSize = ltf.getTableSize();

    // Tolerance thresholds to avoid detecting numerical noise
    const double derivativeThreshold = 1e-6;
    const double secondDerivativeThreshold = 1e-4;

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

        // Only detect sign changes if both derivatives are significant
        if (std::abs(deriv_prev) > derivativeThreshold &&
            std::abs(deriv) > derivativeThreshold &&
            deriv_prev * deriv < 0.0) {  // Sign change = local extremum
            result.localExtrema.push_back(i);

            // Significance = amplitude (how tall the peak/valley is)
            double y = ltf.getBaseLayerValue(i);
            features.push_back({i, std::abs(y), true});
        }
    }

    // 2. Find inflection points (d²y/dx² sign changes) with curvature scores
    for (int i = 2; i < tableSize - 2; ++i) {
        double d2y_prev = estimateSecondDerivative(ltf, i - 1);
        double d2y = estimateSecondDerivative(ltf, i);

        // Only detect sign changes if both second derivatives are significant
        if (std::abs(d2y_prev) > secondDerivativeThreshold &&
            std::abs(d2y) > secondDerivativeThreshold &&
            d2y_prev * d2y < 0.0) {  // Sign change = inflection point
            result.inflectionPoints.push_back(i);

            // Significance = curvature magnitude
            features.push_back({i, std::abs(d2y), false});
        }
    }

    // 3. Prioritize and limit features if maxMandatoryAnchors is set
    result.mandatoryAnchors.push_back(0);  // Always include endpoints
    result.mandatoryAnchors.push_back(tableSize - 1);

    if (maxMandatoryAnchors > 0 && static_cast<int>(features.size()) + 2 > maxMandatoryAnchors) {
        // Too many features - prioritize by significance
        // Reserve 80% for extrema (peaks/valleys are more important than inflection points)
        int maxExtrema = static_cast<int>((maxMandatoryAnchors - 2) * 0.8);
        int maxInflections = (maxMandatoryAnchors - 2) - maxExtrema;

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
    // Central difference (more accurate than forward/backward)
    if (idx == 0 || idx == ltf.getTableSize() - 1)
        return 0.0;  // Boundary handling

    double x0 = ltf.normalizeIndex(idx - 1);
    double x1 = ltf.normalizeIndex(idx + 1);
    double y0 = ltf.getBaseLayerValue(idx - 1);
    double y1 = ltf.getBaseLayerValue(idx + 1);

    return (y1 - y0) / (x1 - x0);
}

double CurveFeatureDetector::estimateSecondDerivative(const LayeredTransferFunction& ltf, int idx) {
    if (idx < 1 || idx >= ltf.getTableSize() - 1)
        return 0.0;

    double h = ltf.normalizeIndex(1) - ltf.normalizeIndex(0);  // Uniform spacing
    double y_prev = ltf.getBaseLayerValue(idx - 1);
    double y = ltf.getBaseLayerValue(idx);
    double y_next = ltf.getBaseLayerValue(idx + 1);

    return (y_next - 2.0 * y + y_prev) / (h * h);
}

} // namespace Services
} // namespace dsp_core
